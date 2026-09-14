#!/usr/bin/env python
"""Live mic -> ten-vad segmenter -> Nemotron streaming ASR, with per-block timing.

Pipeline (see docs/PLAN.md, ADR-001):

    mic ──► capture ──► segmenter ──► ASR ──► final text
           (callback)   (ten-vad)    (MLX)
           int16 16ms    int16        float32
           RT-safe       hop 256      560 ms chunks

Chunk size matters more than the library's 160 ms default suggests.  Measured on
this machine, pushing the same audio contiguously through one session:

    160 ms  -> "We have to think of building England or America or around. The
                second clock in 11 o'clock and thinking about it."
    560 ms  -> "About your clock project. You're doing with your clock project.
                We have to. Think of a building in either London or England ..."
    batch   -> "Me about your clock project, what are you doing with your clock
                project? We have to think of a building in either London or ..."

560 ms is close to batch quality and costs 16% of the realtime budget; 160 ms is
both worse and more expensive (57%), because with x8 subsampling a 160 ms chunk
is only 2 encoder frames of context at zero lookahead.  Hence the 560 ms default.

The audio callback only enqueues; all VAD and ASR work happens on the main
thread so the callback stays real-time-safe.  ten-vad wants int16 at a 160/256
sample hop, the ASR wants float32 in ~160 ms chunks, so the segmenter owns that
conversion and the utterance lifecycle (including StreamSession.reset()).

Timings are reported per block: VAD per frame, ASR per push, plus the
end-of-speech -> final-text latency that the user actually feels.

Usage:
    .venv/bin/python scripts/mic_vad_asr.py
    .venv/bin/python scripts/mic_vad_asr.py --threshold 0.6 --hangover-ms 600
    .venv/bin/python scripts/mic_vad_asr.py --list-devices
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
VAD_HOP = 256  # samples @ 16 kHz = 16 ms; ten-vad supports 160 or 256
INT16_SCALE = 32768.0


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


@dataclass
class Stat:
    """Collects durations in seconds and reports them in ms or us."""

    name: str
    unit: str = "ms"
    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)

    @property
    def n(self) -> int:
        return len(self.samples)

    def row(self) -> str:
        if not self.samples:
            return f"  {self.name:<28} (no samples)"
        scale = 1e6 if self.unit == "us" else 1e3
        a = np.asarray(self.samples) * scale
        return (
            f"  {self.name:<28} n={a.size:<6} "
            f"mean={a.mean():8.2f}  p50={np.percentile(a, 50):8.2f}  "
            f"p90={np.percentile(a, 90):8.2f}  max={a.max():9.2f}  {self.unit}"
        )


class Timers:
    def __init__(self, asr_chunk_ms: int) -> None:
        self._asr_chunk_ms = asr_chunk_ms
        self.vad = Stat("vad.process / frame", unit="us")
        self.asr_push = Stat("asr.push / chunk")
        self.asr_flush = Stat("asr.flush")
        self.asr_reset = Stat("asr.reset")
        self.endpoint_to_final = Stat("end-of-speech -> final")
        self.utterance_total = Stat("utterance wall time")
        self.max_backlog_frames = 0
        self.overruns = 0

    def report(self) -> str:
        lines = ["", "=" * 84, "TIMING SUMMARY (per block)", "=" * 84]
        for s in (
            self.vad,
            self.asr_push,
            self.asr_flush,
            self.asr_reset,
            self.endpoint_to_final,
            self.utterance_total,
        ):
            lines.append(s.row())
        lines.append("")
        backlog_ms = self.max_backlog_frames * VAD_HOP / SAMPLE_RATE * 1000
        lines.append(
            f"  max capture backlog: {self.max_backlog_frames} frames "
            f"({backlog_ms:.0f} ms of audio)"
        )
        lines.append(f"  callback overruns:   {self.overruns}")
        if self.asr_push.n:
            p50_ms = float(np.percentile(np.asarray(self.asr_push.samples), 50)) * 1e3
            lines.append(
                f"  ASR realtime use:    p50 {p50_ms:.0f} ms per "
                f"{self._asr_chunk_ms} ms chunk -> "
                f"{p50_ms / self._asr_chunk_ms * 100:.0f}% of one realtime budget"
            )
        if self.vad.n:
            p50_us = float(np.percentile(np.asarray(self.vad.samples), 50)) * 1e6
            frame_ms = VAD_HOP / SAMPLE_RATE * 1000
            lines.append(
                f"  VAD realtime use:    p50 {p50_us:.0f} us per "
                f"{frame_ms:.0f} ms frame -> {p50_us / 1e3 / frame_ms * 100:.2f}%"
            )
        lines.append("=" * 84)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Segmenter
# ----------------------------------------------------------------------


class Segmenter:
    """ten-vad endpointing state machine.

    Owns the utterance lifecycle.  Takes int16 frames, emits float32 chunks for
    the ASR plus utterance start/end events.  Keeps a short pre-roll so word
    onsets are not clipped when speech is first detected, and rides through
    short internal pauses so sentences are not cut in half.
    """

    def __init__(self, vad, timers: Timers, *, hangover_ms: int,
                 min_utterance_ms: int, preroll_ms: int, asr_chunk_ms: int) -> None:
        self._vad = vad
        self._t = timers
        frame_ms = VAD_HOP / SAMPLE_RATE * 1000
        self._hangover_frames = max(1, round(hangover_ms / frame_ms))
        self._min_utt_frames = max(1, round(min_utterance_ms / frame_ms))
        self._preroll_frames = max(0, round(preroll_ms / frame_ms))
        self._chunk_samples = int(SAMPLE_RATE * asr_chunk_ms / 1000)

        self.reset()

    def reset(self) -> None:
        """Drop all utterance state and start listening from scratch.

        Called between turns by a half-duplex capture gate: audio that arrived
        while the pipeline was answering must not leak into the next utterance,
        and any pre-roll retained from before the gap is stale.

        The TenVad object keeps its own internal feature history across this --
        ten-vad exposes no reset, and re-instantiating it per turn would be
        wasteful.  It re-settles within a few frames of real room tone.
        """
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0
        self._preroll: list[np.ndarray] = []
        self._pending = np.zeros(0, dtype=np.int16)

    def push_frame(self, frame_i16: np.ndarray):
        """Feed one VAD-hop frame.

        Yields ("start", None), ("chunk", float32 array), ("end", n_frames).
        On "end", n_frames is 0 if the utterance was shorter than the minimum.
        """
        t0 = time.perf_counter()
        _prob, flag = self._vad.process(frame_i16)
        self._t.vad.add(time.perf_counter() - t0)

        if not self._in_speech:
            if not flag:
                self._preroll.append(frame_i16.copy())
                if len(self._preroll) > self._preroll_frames:
                    self._preroll.pop(0)
                return
            self._in_speech = True
            self._silence_run = 0
            self._speech_frames = 1
            # prepend pre-roll so the word onset is not clipped
            self._pending = np.concatenate(self._preroll + [frame_i16])
            self._preroll.clear()
            yield ("start", None)
        else:
            # Keep audio even through short internal pauses.
            self._pending = np.concatenate([self._pending, frame_i16])
            self._speech_frames += 1
            self._silence_run = 0 if flag else self._silence_run + 1

        # Emit whole ASR chunks as soon as they are available.
        while len(self._pending) >= self._chunk_samples:
            chunk = self._pending[: self._chunk_samples]
            self._pending = self._pending[self._chunk_samples:]
            yield ("chunk", chunk.astype(np.float32) / INT16_SCALE)

        if self._silence_run >= self._hangover_frames:
            tail, self._pending = self._pending, np.zeros(0, dtype=np.int16)
            if len(tail):
                yield ("chunk", tail.astype(np.float32) / INT16_SCALE)
            n = self._speech_frames
            self._in_speech = False
            self._silence_run = 0
            self._speech_frames = 0
            yield ("end", n if n >= self._min_utt_frames else 0)


# ----------------------------------------------------------------------
# Capture gate
# ----------------------------------------------------------------------


class CaptureGate:
    """Half-duplex microphone gate.

    While the pipeline is answering -- LLM, TTS synthesis, playback -- the
    speakers are audible to the microphone.  Ungated, the VAD flags that audio
    as speech and the pipeline transcribes its own reply, which produces
    another reply, which is heard again: a runaway loop, not a one-off glitch.

    `accept()` is called from the audio callback, so it does nothing but test a
    bool and bump a counter -- cheaper than the queue put it guards.  `open` is
    a plain attribute rather than an Event because attribute assignment is
    atomic under the GIL, and a Lock would put lock acquisition on the audio
    thread for no benefit.

    Closing the gate is not sufficient by itself.  Frames queued *before* it
    closed are still stale; the output device and the room both keep sounding
    briefly after the player process exits; and the segmenter may hold pre-roll
    from before the gap.  reopen() deals with all three, in that order.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.open = True
        self.dropped = 0
        self.closures = 0
        # Where frames go while the gate is shut, if anyone wants them. This is
        # the hook ADR-002 named as the upgrade path -- "accept() becomes
        # 'filter this frame' instead of 'drop it'" -- and barge-in is the first
        # caller. None keeps the original behaviour exactly: dropped on the
        # floor, which is still the default.
        self.listener = None

    def accept(self, frame=None) -> bool:
        """Audio-thread hot path. False means discard this frame.

        `frame` is optional so existing callers are unaffected. When a listener
        is attached it gets a copy of what the gate is suppressing -- a queue
        put, never inference: this runs on the CoreAudio thread and anything
        expensive here is an underrun.
        """
        if self.open:
            return True
        self.dropped += 1
        if self.listener is not None and frame is not None:
            self.listener(frame)
        return False

    def close(self) -> None:
        if not self.enabled:
            return
        self.open = False
        self.closures += 1

    def reopen(self, q=None, seg: "Segmenter | None" = None,
               tail_s: float = 0.25) -> None:
        """Wait out the acoustic tail, discard what accumulated, then listen.

        The drain must happen while the gate is still closed, or the callback
        races it and refills the queue behind us.
        """
        if not self.enabled:
            return
        if tail_s > 0:
            time.sleep(tail_s)
        if q is not None:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if seg is not None:
            seg.reset()
        self.open = True

    @property
    def dropped_s(self) -> float:
        return self.dropped * VAD_HOP / SAMPLE_RATE

    def row(self) -> str:
        if not self.enabled:
            return "  mic gate:            DISABLED (--no-mic-gate)"
        return (f"  mic gate:            {self.closures} closures, "
                f"{self.dropped} frames dropped "
                f"({self.dropped_s:.1f} s never reached the VAD)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def list_devices() -> None:
    import sounddevice as sd

    print("Input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  ch={d['max_input_channels']} "
                  f"sr={int(d['default_samplerate'])}")


def wav_frames(path: str, hangover_ms: int):
    """Yield VAD-hop int16 frames from a 16 kHz mono WAV.

    Used by --wav to exercise the exact same segmenter/ASR path as the mic,
    without needing a live microphone.  Trailing silence is appended so the
    final utterance reaches its endpoint instead of being left open.
    """
    import wave

    with wave.open(path) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(
                f"--wav needs 16 kHz mono PCM16; got {w.getframerate()} Hz, "
                f"{w.getnchannels()} ch, {w.getsampwidth() * 8} bit"
            )
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    for i in range(len(pcm) // VAD_HOP):
        yield pcm[i * VAD_HOP:(i + 1) * VAD_HOP]
    # drain: enough silence to trip the hangover and finalise the last utterance
    pad = int(hangover_ms / (VAD_HOP / SAMPLE_RATE * 1000)) + 2
    silence = np.zeros(VAD_HOP, dtype=np.int16)
    for _ in range(pad):
        yield silence


def main(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import sounddevice as sd
    from ten_vad import TenVad

    from nemotron_asr_mlx import from_pretrained

    timers = Timers(args.asr_chunk_ms)
    chunk_samples = int(SAMPLE_RATE * args.asr_chunk_ms / 1000)

    # --- load and warm the ASR: the first push pays a ~3 s Metal kernel compile,
    #     so pay it now rather than losing the user's first word ---
    print("loading ASR model ...", flush=True)
    t0 = time.perf_counter()
    model = from_pretrained(args.model)
    print(f"  model load:   {time.perf_counter() - t0:6.2f} s")

    t0 = time.perf_counter()
    warm = model.create_stream(chunk_ms=args.asr_chunk_ms)
    warm.push(mx.zeros(chunk_samples))
    mx.eval()
    print(f"  warm-up push: {time.perf_counter() - t0:6.2f} s  "
          f"(Metal kernel compile, paid up front)")
    del warm

    vad = TenVad(VAD_HOP, args.threshold)
    seg = Segmenter(
        vad, timers,
        hangover_ms=args.hangover_ms,
        min_utterance_ms=args.min_utterance_ms,
        preroll_ms=args.preroll_ms,
        asr_chunk_ms=args.asr_chunk_ms,
    )
    session = model.create_stream(chunk_ms=args.asr_chunk_ms)

    print(f"\nVAD hop {VAD_HOP} samples ({VAD_HOP / SAMPLE_RATE * 1000:.0f} ms), "
          f"threshold {args.threshold}, hangover {args.hangover_ms} ms, "
          f"ASR chunk {args.asr_chunk_ms} ms")

    stream = None
    if args.wav:
        frame_source = wav_frames(args.wav, args.hangover_ms)
        print(f"source: {args.wav} (offline replay through the live path)\n")
    else:
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, time_info, status):  # audio thread: enqueue only
            if status:
                timers.overruns += 1
            q.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=VAD_HOP, device=args.device, callback=callback,
        )

        def mic_frames():
            while True:
                try:
                    f = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                timers.max_backlog_frames = max(timers.max_backlog_frames, q.qsize())
                yield f

        frame_source = mic_frames()
        print("listening — speak, then pause. Ctrl-C to stop.\n")

    utt_start = 0.0
    last_chunk_done = 0.0
    n_pushes = 0
    utt_asr_s = 0.0
    n_utt = 0

    if stream is not None:
        stream.start()
    t_run = time.perf_counter()
    try:
        for frame in frame_source:
            for kind, payload in seg.push_frame(frame):
                if kind == "start":
                    utt_start = time.perf_counter()
                    n_pushes = 0
                    utt_asr_s = 0.0
                    print("  [speech] ", end="", flush=True)

                elif kind == "chunk":
                    t = time.perf_counter()
                    ev = session.push(mx.array(payload))
                    mx.eval()
                    dt = time.perf_counter() - t
                    timers.asr_push.add(dt)
                    utt_asr_s += dt
                    n_pushes += 1
                    last_chunk_done = time.perf_counter()
                    if ev.text_delta:
                        print(ev.text_delta, end="", flush=True)

                elif kind == "end":
                    t = time.perf_counter()
                    final = session.flush()
                    timers.asr_flush.add(time.perf_counter() - t)
                    t = time.perf_counter()
                    session.reset()
                    timers.asr_reset.add(time.perf_counter() - t)

                    if payload == 0:  # shorter than --min-utterance-ms
                        print(" (too short, discarded)")
                        continue

                    n_utt += 1
                    wall = time.perf_counter() - utt_start
                    timers.utterance_total.add(wall)
                    if last_chunk_done:
                        timers.endpoint_to_final.add(
                            time.perf_counter() - last_chunk_done)
                    speech_s = payload * VAD_HOP / SAMPLE_RATE

                    print()
                    print(f"  >>> [{n_utt}] {final.text.strip() or '(empty)'}")
                    print(f"      speech {speech_s:.2f}s | {n_pushes} pushes | "
                          f"asr {utt_asr_s * 1e3:.0f}ms "
                          f"({utt_asr_s / max(speech_s, 1e-6) * 100:.0f}% of realtime) | "
                          f"wall {wall:.2f}s\n")

    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        print(timers.report())
        print(f"  total run time:      {time.perf_counter() - t_run:.2f} s")
        print(f"\nutterances transcribed: {n_utt}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live mic -> ten-vad -> Nemotron ASR with per-block timing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list-devices", action="store_true",
                   help="list input devices and exit")
    p.add_argument("--device", default=None, help="input device index or name")
    p.add_argument("--wav", default=None,
                   help="replay a 16 kHz mono PCM16 WAV through the same "
                        "segmenter/ASR path instead of the mic (for testing)")
    p.add_argument("--model", default="dboris/nemotron-asr-mlx")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="ten-vad speech threshold (default 0.5)")
    p.add_argument("--hangover-ms", type=int, default=500,
                   help="trailing silence needed to end an utterance (default 500)")
    p.add_argument("--min-utterance-ms", type=int, default=200,
                   help="discard utterances shorter than this (default 200)")
    p.add_argument("--preroll-ms", type=int, default=128,
                   help="audio retained before speech onset (default 128)")
    p.add_argument("--asr-chunk-ms", type=int, default=560,
                   help="ASR chunk size, multiple of 80 ms (default 560). "
                        "Measured: 160 ms badly degrades accuracy AND costs 57%% "
                        "of the realtime budget; 560 ms is near-batch quality at 16%%")
    a = p.parse_args()
    if a.asr_chunk_ms % 80:
        p.error("--asr-chunk-ms must be a multiple of 80 "
                "(encoder subsamples 10 ms mel frames by 8)")
    if a.device is not None and str(a.device).isdigit():
        a.device = int(a.device)
    return a


if __name__ == "__main__":
    _args = parse_args()
    if _args.list_devices:
        list_devices()
        sys.exit(0)
    sys.exit(main(_args))
