#!/usr/bin/env python
"""Mic -> ten-vad -> Nemotron ASR -> small LLM correction -> edge-tts, fully timed.

Stages, and what each costs, are reported per turn and as a summary waterfall:

    mic ──► capture ──► segmenter ──► ASR ──► LLM ──► edge-tts ──► playback
           (callback)   (ten-vad)    (MLX)   (MLX)   (cloud)      (ffplay)

The VAD + ASR half is imported from mic_vad_asr.py so there is one source of
truth for the segmenter; that script is left untouched and still runs on its own.

Two things worth knowing about the latency profile before reading the numbers:

* edge-tts is a **network** service.  Measured on this machine it needs ~690 ms
  to return its first audio chunk, which is the single largest term in the whole
  pipeline -- larger than ASR, LLM and VAD combined.  It is also the one stage
  that breaks the offline property the rest of the stack has.
* The LLM and the ASR share the same 8 GPU cores, so they contend.  Here they
  never overlap, because a turn is strictly sequential: the user stops talking,
  then we correct, then we speak.  The breakdown below is therefore
  contention-free -- which stops being true the moment you add barge-in.

Usage:
    .venv/bin/python scripts/voice_correct.py
    .venv/bin/python scripts/voice_correct.py --voice en-GB-SoniaNeural
    .venv/bin/python scripts/voice_correct.py --wav clip.wav --player none
    .venv/bin/python scripts/voice_correct.py --list-devices
"""

from __future__ import annotations

import argparse
import asyncio
import os
import queue
import re
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mic_vad_asr import (  # noqa: E402  (sys.path set immediately above)
    SAMPLE_RATE,
    VAD_HOP,
    Segmenter,
    Stat,
    list_devices,
    wav_frames,
)

SYSTEM_PROMPT = (
    "You clean up speech-to-text output. Fix grammar, punctuation, "
    "capitalisation and obvious transcription errors. Keep the speaker's "
    "meaning and wording wherever possible - do not add, explain or answer "
    "anything. Reply with ONLY the corrected text."
)

ASSISTANT_PROMPT = (
    "You are a helpful voice assistant. What you read as the user's message "
    "is speech transcribed to text, so it may contain minor errors - infer "
    "intent rather than pointing them out. Reply the way you would speak "
    "out loud: plain sentences, no markdown, no bullet lists, no headings. "
    "Keep replies to one to three short sentences. If a question rests on a "
    "false premise, say so plainly instead of playing along."
)

# Splits streamed text into sentences as soon as a boundary appears, so the
# caller can start TTS on sentence one while generation continues.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


@dataclass
class TurnTiming:
    """Every latency term for a single turn, in seconds."""

    speech: float = 0.0
    asr_push: float = 0.0
    asr_final: float = 0.0
    llm_ttft: float = 0.0
    llm_total: float = 0.0
    tts_first: float = 0.0
    tts_total: float = 0.0
    play_start: float = 0.0
    llm_prompt_tokens: int = 0
    llm_gen_tokens: int = 0
    llm_prompt_tps: float = 0.0  # prompt eval rate, tok/s
    llm_gen_tps: float = 0.0     # decode rate, tok/s (excludes prompt eval)

    @property
    def response(self) -> float:
        """End of speech -> first audio out. The latency the user actually feels."""
        return self.asr_final + self.llm_total + self.tts_first + self.play_start

    @property
    def total(self) -> float:
        return self.asr_final + self.llm_total + self.tts_total + self.play_start


class Timers:
    """Aggregates per-stage stats. Exposes `.vad` so Segmenter can record into it."""

    def __init__(self, asr_chunk_ms: int) -> None:
        self._chunk_ms = asr_chunk_ms
        self.vad = Stat("vad.process / frame", unit="us")
        self.asr_push = Stat("asr.push / chunk")
        self.turns: list[TurnTiming] = []
        self.overruns = 0
        self.max_backlog = 0

    def _agg(self, name: str, fn) -> Stat:
        s = Stat(name)
        for t in self.turns:
            s.add(fn(t))
        return s

    def report(self) -> str:
        L = ["", "=" * 88, "LATENCY BREAKDOWN", "=" * 88, ""]
        L.append("Streaming stages (these run while the user is still talking, so they")
        L.append("cost throughput, not response latency):")
        L.append(self.vad.row())
        L.append(self.asr_push.row())
        if self.asr_push.n:
            p50 = float(np.percentile(np.asarray(self.asr_push.samples), 50)) * 1e3
            L.append(f"      -> {p50:.0f} ms per {self._chunk_ms} ms chunk = "
                     f"{p50 / self._chunk_ms * 100:.0f}% of one realtime budget")

        if not self.turns:
            L.append("")
            L.append("  (no completed turns)")
            L.append("=" * 88)
            return "\n".join(L)

        L.append("")
        L.append("Response path (after the user stops talking -- this IS the felt latency):")
        for name, fn in (
            ("asr.flush + reset", lambda t: t.asr_final),
            ("llm time-to-first-token", lambda t: t.llm_ttft),
            ("llm total generation", lambda t: t.llm_total),
            ("edge-tts to first audio", lambda t: t.tts_first),
            ("edge-tts total synthesis", lambda t: t.tts_total),
            ("player spawn", lambda t: t.play_start),
            ("END-OF-SPEECH -> AUDIO OUT", lambda t: t.response),
        ):
            L.append(self._agg(name, fn).row())

        med = {
            "ASR finalize": float(np.median([t.asr_final for t in self.turns])),
            "LLM generate": float(np.median([t.llm_total for t in self.turns])),
            "edge-tts (network)": float(np.median([t.tts_first for t in self.turns])),
            "player spawn": float(np.median([t.play_start for t in self.turns])),
        }
        total = sum(med.values()) or 1e-9
        L.append("")
        L.append(f"Where the median {total * 1e3:.0f} ms of response latency goes:")
        for k, v in med.items():
            bar = "#" * max(0, round(v / total * 52))
            L.append(f"  {k:<20} {v * 1e3:7.0f} ms  {v / total * 100:5.1f}%  {bar}")

        L.append("")
        L.append("  Note: the VAD hangover is charged before any of this -- the pipeline")
        L.append(f"  cannot know the turn ended until {HANGOVER_NOTE[0]} ms of silence has passed,")
        L.append(f"  so wall-clock from last word to audio is ~{HANGOVER_NOTE[0] + total * 1e3:.0f} ms.")

        if self.turns:
            L.append("")
            L.append(
                f"  LLM detail: {np.mean([t.llm_prompt_tokens for t in self.turns]):.0f} prompt tok "
                f"@ {np.mean([t.llm_prompt_tps for t in self.turns]):.0f} tok/s eval, "
                f"{np.mean([t.llm_gen_tokens for t in self.turns]):.0f} gen tok "
                f"@ {np.mean([t.llm_gen_tps for t in self.turns]):.1f} tok/s decode"
            )
            L.append("  (TTFT is dominated by prompt eval + first-token overhead, not decode "
                     "speed;")
            L.append("   a shorter system prompt is the cheapest way to cut it.)")
        L.append("")
        L.append(f"  max capture backlog: {self.max_backlog} frames | "
                 f"callback overruns: {self.overruns}")
        L.append("=" * 88)
        return "\n".join(L)


HANGOVER_NOTE = [800]  # filled from args so the report can mention it


# ----------------------------------------------------------------------
# LLM correction
# ----------------------------------------------------------------------


class Corrector:
    def __init__(self, repo: str, max_tokens: int, temp: float,
                 use_cache: bool = True, mode: str = "correct",
                 history_turns: int = 6) -> None:
        from mlx_lm import load

        self._max_tokens = max_tokens
        self._temp = temp
        self._use_cache = use_cache
        self._mode = mode
        self._history_turns = history_turns
        self.truncated = False
        t0 = time.perf_counter()
        self._model, self._tok = load(repo)
        self.load_s = time.perf_counter() - t0

        # Assist-mode only: bounded conversation history and a growing KV
        # cache. Unused (and untouched) by mode="correct".
        self._history: list[dict] = []
        self._cache = None
        self._cache_len = 0
        if self._mode == "assist" and self._use_cache:
            self._reset_cache()

    def _reset_cache(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self._cache = make_prompt_cache(self._model)
        self._cache_len = 0

    def _prompt(self, text: str) -> str:
        return self._tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def correct(self, text: str, timing: TurnTiming) -> str:
        """Stream a correction, recording time-to-first-token and total time."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=self._temp)
        t0 = time.perf_counter()
        first = None
        parts: list[str] = []
        last = None
        for r in stream_generate(self._model, self._tok, self._prompt(text),
                                 max_tokens=self._max_tokens, sampler=sampler):
            if first is None:
                first = time.perf_counter() - t0
            parts.append(r.text)
            last = r
        timing.llm_ttft = first or 0.0
        timing.llm_total = time.perf_counter() - t0
        if last is not None:
            timing.llm_prompt_tokens = last.prompt_tokens
            timing.llm_gen_tokens = last.generation_tokens
            timing.llm_prompt_tps = last.prompt_tps
            timing.llm_gen_tps = last.generation_tps
        return _tidy("".join(parts))

    # ---- assist mode ---------------------------------------------------
    # `mode="assist"` answers the user instead of just correcting the
    # transcript. It keeps a bounded conversation history and, when
    # use_cache is set, a single growing KV cache: each turn only the
    # tokens past the previously-cached prefix are fed to the model, and
    # the assistant's own reply stays resident so the next turn's "history
    # so far" prefix is a cache hit too. When `history_turns` makes the
    # window slide (the oldest exchange is dropped), tokens are removed
    # from the middle rather than the end, so that turn rebuilds the cache
    # from scratch instead of trimming it -- a deliberate simplicity/perf
    # trade for long conversations, not a correctness issue.

    def _assist_messages(self, text: str) -> list[dict]:
        return ([{"role": "system", "content": ASSISTANT_PROMPT}]
                + self._history
                + [{"role": "user", "content": text}])

    def respond_stream(self, text: str, turn):
        """Stream a conversational reply, yielding it sentence by sentence.

        `turn` gains the same fields `correct()` sets, plus
        `llm_first_sentence`; `self.truncated` reports whether the reply
        was cut off at `max_tokens`.
        """
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        full_tokens = self._tok.apply_chat_template(
            self._assist_messages(text), add_generation_prompt=True,
            tokenize=True,
        )

        prompt = full_tokens
        prompt_cache = None
        if self._use_cache:
            if self._cache is None:
                self._reset_cache()
            new_tokens = full_tokens[self._cache_len:]
            if new_tokens:
                prompt, prompt_cache = new_tokens, self._cache
            else:
                # Exact cache hit: stream_generate can't take an empty
                # prompt. Rebuild rather than special-case a back-off.
                self._reset_cache()

        sampler = make_sampler(temp=self._temp)
        t0 = time.perf_counter()
        first = None
        buf = ""
        said_parts: list[str] = []
        last = None
        for r in stream_generate(self._model, self._tok, prompt,
                                 max_tokens=self._max_tokens, sampler=sampler,
                                 prompt_cache=prompt_cache):
            if first is None:
                first = time.perf_counter() - t0
            buf += r.text
            last = r
            while True:
                m = _SENTENCE_BOUNDARY.search(buf)
                if not m:
                    break
                sentence, buf = buf[:m.start()].strip(), buf[m.end():]
                if sentence:
                    if turn.llm_first_sentence == 0.0:
                        turn.llm_first_sentence = time.perf_counter() - t0
                    said_parts.append(sentence)
                    yield sentence
        tail = buf.strip()
        if tail:
            if turn.llm_first_sentence == 0.0:
                turn.llm_first_sentence = time.perf_counter() - t0
            said_parts.append(tail)
            yield tail

        turn.llm_ttft = first or 0.0
        turn.llm_total = time.perf_counter() - t0
        if last is not None:
            turn.llm_prompt_tokens = last.prompt_tokens
            turn.llm_gen_tokens = last.generation_tokens
            turn.llm_prompt_tps = last.prompt_tps
            turn.llm_gen_tps = last.generation_tps
            self.truncated = last.finish_reason == "length"
            if self._use_cache:
                self._cache_len = len(full_tokens) + last.generation_tokens

        reply = " ".join(said_parts)
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": reply})
        if self._history_turns <= 0:
            self._history = []
            if self._use_cache:
                self._reset_cache()
        elif len(self._history) > 2 * self._history_turns:
            self._history = self._history[-2 * self._history_turns:]
            if self._use_cache:
                self._reset_cache()

    def respond(self, text: str, turn) -> str:
        """Non-streaming convenience wrapper, used for warm-up."""
        return " ".join(self.respond_stream(text, turn))

    def reset_history(self) -> None:
        """Drop conversation history (and the cache built on top of it)."""
        self._history = []
        if self._use_cache:
            self._reset_cache()


def _tidy(s: str) -> str:
    """Strip the wrappers small instruct models like to add."""
    s = s.strip()
    for pre in ("Corrected text:", "Corrected:", "Output:"):
        if s.lower().startswith(pre.lower()):
            s = s[len(pre):].strip()
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


# ----------------------------------------------------------------------
# edge-tts + playback
# ----------------------------------------------------------------------


class Speaker:
    def __init__(self, voice: str, player: str) -> None:
        self._voice = voice
        self._player = player

    def _spawn(self):
        if self._player == "ffplay":
            return subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        return None

    async def _stream(self, text: str, timing: TurnTiming) -> int:
        import edge_tts

        t0 = time.perf_counter()
        proc = None
        chunks: list[bytes] = []
        nbytes = 0
        comm = edge_tts.Communicate(text, self._voice)
        async for ch in comm.stream():
            if ch["type"] != "audio":
                continue
            data = ch["data"]
            nbytes += len(data)
            if timing.tts_first == 0.0:
                timing.tts_first = time.perf_counter() - t0
                ts = time.perf_counter()
                proc = self._spawn()
                timing.play_start = time.perf_counter() - ts
            if proc is not None and proc.stdin:
                try:
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    proc = None
            else:
                chunks.append(data)
        timing.tts_total = time.perf_counter() - t0

        if proc is not None and proc.stdin:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            proc.wait()
        elif chunks and self._player == "afplay":
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(b"".join(chunks))
                path = f.name
            subprocess.run(["afplay", path], check=False)
            os.unlink(path)
        return nbytes

    def speak(self, text: str, timing: TurnTiming) -> int:
        return asyncio.run(self._stream(text, timing))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import sounddevice as sd
    from ten_vad import TenVad

    from nemotron_asr_mlx import from_pretrained

    HANGOVER_NOTE[0] = args.hangover_ms
    timers = Timers(args.asr_chunk_ms)
    chunk_samples = int(SAMPLE_RATE * args.asr_chunk_ms / 1000)

    print("loading models ...", flush=True)
    t0 = time.perf_counter()
    asr = from_pretrained(args.model)
    print(f"  ASR load:     {time.perf_counter() - t0:6.2f} s")

    t0 = time.perf_counter()
    warm = asr.create_stream(chunk_ms=args.asr_chunk_ms)
    warm.push(mx.zeros(chunk_samples))
    mx.eval()
    print(f"  ASR warm-up:  {time.perf_counter() - t0:6.2f} s  (Metal compile, paid up front)")
    del warm

    corrector = Corrector(args.llm, args.max_tokens, args.temp)
    print(f"  LLM load:     {corrector.load_s:6.2f} s  ({args.llm})")

    t0 = time.perf_counter()
    corrector.correct("this are a warmup sentence", TurnTiming())
    print(f"  LLM warm-up:  {time.perf_counter() - t0:6.2f} s")

    speaker = Speaker(args.voice, args.player)

    vad = TenVad(VAD_HOP, args.threshold)
    seg = Segmenter(
        vad, timers,
        hangover_ms=args.hangover_ms,
        min_utterance_ms=args.min_utterance_ms,
        preroll_ms=args.preroll_ms,
        asr_chunk_ms=args.asr_chunk_ms,
    )
    session = asr.create_stream(chunk_ms=args.asr_chunk_ms)

    stream = None
    if args.wav:
        frame_source = wav_frames(args.wav, args.hangover_ms)
        print(f"\nsource: {args.wav} (offline replay through the live path)")
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
                timers.max_backlog = max(timers.max_backlog, q.qsize())
                yield f

        frame_source = mic_frames()

    print(f"\nVAD hop {VAD_HOP / SAMPLE_RATE * 1000:.0f} ms, threshold {args.threshold}, "
          f"hangover {args.hangover_ms} ms, ASR chunk {args.asr_chunk_ms} ms, "
          f"voice {args.voice}")
    print("speak, then pause. Ctrl-C to stop.\n")

    turn = TurnTiming()
    n_pushes = 0
    n_turn = 0

    if stream is not None:
        stream.start()
    try:
        for frame in frame_source:
            for kind, payload in seg.push_frame(frame):
                if kind == "start":
                    turn = TurnTiming()
                    n_pushes = 0
                    print("  [speech] ", end="", flush=True)

                elif kind == "chunk":
                    t = time.perf_counter()
                    ev = session.push(mx.array(payload))
                    mx.eval()
                    dt = time.perf_counter() - t
                    timers.asr_push.add(dt)
                    turn.asr_push += dt
                    n_pushes += 1
                    if ev.text_delta:
                        print(ev.text_delta, end="", flush=True)

                elif kind == "end":
                    t = time.perf_counter()
                    final = session.flush()
                    session.reset()
                    turn.asr_final = time.perf_counter() - t

                    if payload == 0:
                        print(" (too short, discarded)")
                        continue
                    raw = final.text.strip()
                    if not raw:
                        print(" (empty transcript, skipped)")
                        continue

                    turn.speech = payload * VAD_HOP / SAMPLE_RATE
                    n_turn += 1
                    print()
                    print(f"  ASR  [{n_turn}]: {raw}")

                    fixed = corrector.correct(raw, turn)
                    print(f"  LLM  [{n_turn}]: {fixed}")

                    try:
                        nbytes = speaker.speak(fixed, turn)
                    except Exception as e:  # network / voice / player failure
                        print(f"  TTS  [{n_turn}]: FAILED ({type(e).__name__}: {e})")
                        continue

                    timers.turns.append(turn)
                    print(f"  ---- turn {n_turn}: speech {turn.speech:.2f}s | "
                          f"{n_pushes} pushes, asr {turn.asr_push * 1e3:.0f}ms | "
                          f"llm {turn.llm_total * 1e3:.0f}ms "
                          f"(ttft {turn.llm_ttft * 1e3:.0f}ms, {turn.llm_gen_tokens} tok) | "
                          f"tts {turn.tts_first * 1e3:.0f}ms first / "
                          f"{turn.tts_total * 1e3:.0f}ms all ({nbytes}B)")
                    print(f"       END-OF-SPEECH -> AUDIO OUT: {turn.response * 1e3:.0f} ms\n")

    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        print(timers.report())
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mic -> VAD -> ASR -> LLM correction -> edge-tts, with latency breakdown.",
    )
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device", default=None, help="input device index or name")
    p.add_argument("--wav", default=None,
                   help="replay a 16 kHz mono PCM16 WAV instead of the mic")
    p.add_argument("--model", default="dboris/nemotron-asr-mlx", help="ASR model")
    p.add_argument("--llm", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                   help="correction LLM (mlx-community repo)")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temp", type=float, default=0.0, help="0 = greedy, most repeatable")
    p.add_argument("--voice", default="en-US-AriaNeural", help="edge-tts voice")
    p.add_argument("--player", default="ffplay", choices=("ffplay", "afplay", "none"),
                   help="ffplay streams chunks as they arrive (lowest latency)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hangover-ms", type=int, default=800)
    p.add_argument("--min-utterance-ms", type=int, default=300)
    p.add_argument("--preroll-ms", type=int, default=128)
    p.add_argument("--asr-chunk-ms", type=int, default=560,
                   help="multiple of 80 ms; 560 measured near-batch quality")
    a = p.parse_args()
    if a.asr_chunk_ms % 80:
        p.error("--asr-chunk-ms must be a multiple of 80")
    if a.device is not None and str(a.device).isdigit():
        a.device = int(a.device)
    return a


if __name__ == "__main__":
    _args = parse_args()
    if _args.list_devices:
        list_devices()
        sys.exit(0)
    sys.exit(main(_args))
