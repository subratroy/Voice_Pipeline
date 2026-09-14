#!/usr/bin/env python
"""Production-shaped voice loop: mic -> VAD -> ASR -> LLM -> streaming TTS -> speakers.

The Chatterbox build (voice_chatterbox.py) spent 89% of its response latency in
TTS -- 9519 ms median, because Chatterbox has no streaming generate, so
time-to-first-audio *was* full synthesis and grew with reply length.  This script
replaces that with four independent changes, each measured (docs/PLAN.md ADR-003):

    1. pocket-tts on MLX in place of Chatterbox    9519 ms -> ~250 ms TTFA
    2. streaming playback, chunk by chunk           playback starts on chunk 1
    3. mlx-lm prompt caching for the system prompt   733 ms -> 257 ms TTFT
    4. a shorter VAD hangover  800 ms -> 600 ms       (450 ms fragments; measured)

Everything runs on MLX against numpy 2.5.2 -- ASR, LLM and TTS in one process,
one venv, no torch.  Chatterbox needed a second venv, a subprocess worker and a
JSON pipe purely because it pinned numpy<2.

The half-duplex capture gate from ADR-002 is preserved, with one change that
matters: it reopens when the *output queue has drained*, not when synthesis
returned.  Reopening early would put the tail of our own reply back into the
microphone.

Usage:
    .venv/bin/python scripts/voice_live.py
    .venv/bin/python scripts/voice_live.py --wav clip.wav --player none
    .venv/bin/python scripts/voice_live.py --engine pocket --voice marius
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mic_vad_asr import (  # noqa: E402
    SAMPLE_RATE,
    VAD_HOP,
    CaptureGate,
    Segmenter,
    Stat,
    list_devices,
    wav_frames,
)
from tts_engines import KNOWN, StreamingPlayer, open_engine  # noqa: E402
from voice_correct import Corrector  # noqa: E402  (carries SYSTEM_PROMPT)


@dataclass
class Turn:
    speech: float = 0.0
    asr_push: float = 0.0
    asr_final: float = 0.0
    llm_ttft: float = 0.0
    llm_first_sentence: float = 0.0   # first speakable sentence closed
    llm_total: float = 0.0
    truncated: bool = False
    tts_ttfa: float = 0.0      # request -> first audio chunk out of the engine
    tts_synth: float = 0.0     # request -> last chunk
    tts_audio_s: float = 0.0
    play_dur: float = 0.0      # reply sounding; NOT part of response latency
    underruns: int = 0
    llm_prompt_tokens: int = 0
    llm_gen_tokens: int = 0
    llm_prompt_tps: float = 0.0
    llm_gen_tps: float = 0.0

    @property
    def response(self) -> float:
        """End of speech -> first audio leaves the speaker. The felt latency.

        Uses llm_first_sentence, not llm_total: TTS starts on sentence one, so
        the rest of generation overlaps playback and never reaches the listener
        as waiting. Full generation stays in the report separately so this build
        remains comparable with the pre-streaming one.
        """
        return self.asr_final + (self.llm_first_sentence or self.llm_total) \
            + self.tts_ttfa

    @property
    def rtf(self) -> float:
        return self.tts_synth / self.tts_audio_s if self.tts_audio_s else 0.0


class Timers:
    def __init__(self, asr_chunk_ms: int) -> None:
        self._chunk_ms = asr_chunk_ms
        self.vad = Stat("vad.process / frame", unit="us")
        self.asr_push = Stat("asr.push / chunk")
        self.turns: list[Turn] = []
        self.overruns = 0
        self.max_backlog = 0
        self.gate: CaptureGate | None = None
        self.barge = None

    def _agg(self, name, fn) -> Stat:
        s = Stat(name)
        for t in self.turns:
            s.add(fn(t))
        return s

    def report(self, hangover_ms: int, engine: str) -> str:
        L = ["", "=" * 88, f"LATENCY BREAKDOWN  ({engine}, streaming, all-MLX)",
             "=" * 88, "",
             "Streaming stages (run while the user is still talking, so free):",
             self.vad.row(), self.asr_push.row()]
        if not self.turns:
            L += ["", "  (no completed turns)", "=" * 88]
            return "\n".join(L)

        L += ["", "Response path (after the user stops talking -- the felt latency):"]
        for name, fn in (
            ("asr.flush + reset", lambda t: t.asr_final),
            ("llm time-to-first-token", lambda t: t.llm_ttft),
            ("llm first sentence", lambda t: t.llm_first_sentence),
            ("(llm total generation)", lambda t: t.llm_total),
            ("tts time-to-first-audio", lambda t: t.tts_ttfa),
            ("END-OF-SPEECH -> AUDIO OUT", lambda t: t.response),
            ("(tts full synthesis)", lambda t: t.tts_synth),
            ("(playback duration)", lambda t: t.play_dur),
        ):
            L.append(self._agg(name, fn).row())

        med = {"ASR finalize": np.median([t.asr_final for t in self.turns]),
               "LLM first sentence": np.median(
                   [t.llm_first_sentence or t.llm_total for t in self.turns]),
               "TTS first audio": np.median([t.tts_ttfa for t in self.turns])}
        total = float(sum(med.values())) or 1e-9
        L += ["", f"Where the median {total * 1e3:.0f} ms of response latency goes:"]
        for k, v in med.items():
            L.append(f"  {k:<20} {v * 1e3:8.0f} ms  {v / total * 100:5.1f}%  "
                     f"{'#' * max(0, round(v / total * 50))}")

        rtf = [t.rtf for t in self.turns if t.rtf]
        if rtf:
            L += ["", f"  TTS RTF: median {np.median(rtf):.2f}x "
                      f"({'faster' if np.median(rtf) < 1 else 'SLOWER'} than realtime)"]
        ntrunc = sum(1 for t in self.turns if t.truncated)
        if ntrunc:
            L.append(f"  replies cut off at --max-tokens: {ntrunc}/{len(self.turns)}"
                     "   <-- raise it; TTS is speaking fragments")
        under = sum(t.underruns for t in self.turns)
        L.append(f"  playback underruns: {under}"
                 + ("" if under == 0 else "   <-- engine slower than realtime"))
        L += ["", f"  Plus the {hangover_ms} ms VAD hangover before all of it: "
                  f"wall-clock ~{hangover_ms + total * 1e3:.0f} ms from last word.",
              "", f"  LLM: {np.mean([t.llm_prompt_tokens for t in self.turns]):.0f} prompt tok "
                  f"@ {np.mean([t.llm_prompt_tps for t in self.turns]):.0f} t/s eval, "
                  f"{np.mean([t.llm_gen_tokens for t in self.turns]):.0f} gen tok "
                  f"@ {np.mean([t.llm_gen_tps for t in self.turns]):.1f} t/s decode",
              "", f"  max capture backlog: {self.max_backlog} frames | "
                  f"callback overruns: {self.overruns}"]
        if self.gate is not None:
            L.append(self.gate.row())
        if getattr(self, "barge", None) is not None:
            L.append(self.barge.stats.row())
        L.append("=" * 88)
        return "\n".join(L)


def speak(engine, player, sentences, turn: Turn, lang: str, voice,
          instruct, barge=None) -> str:
    """Synthesise sentence by sentence, playing each as soon as it is ready.

    `sentences` is an iterator, so the LLM is still generating while earlier
    sentences are being synthesised and played. tts_ttfa is measured to the
    first audio chunk of the first sentence -- that is what the listener waits
    through; everything after it overlaps playback.
    """
    n = 0
    said: list[str] = []
    synth = 0.0
    t_first: float | None = None
    for sentence in sentences:
        if barge is not None and barge.stopped:
            # Abandoning the LLM generator here is what actually stops
            # generation -- it is a generator, so not pulling from it again
            # ends it. Breaking before appending also keeps `said` to what was
            # really spoken, which is what goes into the conversation history:
            # recording text the user cut off would have the next turn's context
            # claim the assistant said things it never got to say.
            break
        if not sentence:
            continue
        # Clock starts when the sentence is IN HAND. Starting it before pulling
        # from the generator would fold the LLM's generation time into
        # tts_ttfa, and `response` already counts that as llm_first_sentence --
        # double-counting roughly 475 ms per turn.
        t_s = time.perf_counter()
        if t_first is None:
            t_first = t_s
        said.append(sentence)
        for chunk in engine.stream(sentence, lang=lang, voice=voice,
                                   instruct=instruct):
            if barge is not None and barge.stopped:
                break
            if turn.tts_ttfa == 0.0:
                turn.tts_ttfa = time.perf_counter() - t_s
            n += chunk.size
            if player is not None:
                player.feed(chunk)
        synth += time.perf_counter() - t_s
    turn.tts_synth = synth
    turn.tts_audio_s = n / engine.sr
    if player is not None:
        tp = time.perf_counter()
        player.finish()
        # A stopped reply has already had its queue dropped, so wait() returns
        # at once and the capture gate reopens now rather than after audio
        # nobody is listening to.
        player.wait()
        turn.play_dur = time.perf_counter() - tp
        turn.underruns = player.underruns
    return " ".join(said)


def _echo_probe(sd, device) -> tuple[bool, float]:
    """Play a chirp and listen for it. (audible, correlation).

    Deliberately a measurement rather than a device-name heuristic: "MacBook Pro
    Speakers" vs "AirPods" guesses wrong for external DACs, and the thing that
    actually matters is whether this room couples output to input right now.
    """
    import numpy as np

    from bargein import echo_check
    from mic_vad_asr import SAMPLE_RATE

    def playrec(samples):
        # playrec, not play-then-rec: sd.play and sd.rec share one stream, so
        # recording after playing cancels the playback and the probe measures
        # silence -- reporting "safe" on speakers, which is the one answer it
        # must never get wrong.
        rec = sd.playrec(samples, samplerate=SAMPLE_RATE, channels=1,
                         dtype="float32", device=device)
        sd.wait()
        return rec[:, 0]

    try:
        return echo_check(playrec)
    except Exception as e:
        print(f"  echo check unavailable ({type(e).__name__})")
        return "inconclusive", float("nan")


def main(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import sounddevice as sd
    from ten_vad import TenVad

    from nemotron_asr_mlx import from_pretrained

    timers = Timers(args.asr_chunk_ms)
    gate = CaptureGate(enabled=not args.no_mic_gate)
    timers.gate = gate

    # Barge-in, opt-in. Off, ADR-002's half-duplex behaviour is exactly as it
    # was: the gate drops suppressed frames and nothing listens during a reply.
    barge = vprint = embedder = None
    if args.barge_in:
        from bargein import BargeInDetector, BargeInRunner
        from speaker_id import SpeakerEmbedder, Voiceprint

        embedder = SpeakerEmbedder(args.speaker_model)
        vprint = Voiceprint()
        det = BargeInDetector(vprint, embedder, threshold=args.barge_threshold)

        def _duck(d):
            if player is not None:
                player.duck(args.duck_level)

        def _resume(d):
            if player is not None:
                player.duck(1.0)
            if d.keyword:                 # a hit that did not survive the gate
                print(d.row())

        def _stop(d):
            if player is not None:
                player.stop()
            print(d.row())

        barge = BargeInRunner(det, on_duck=_duck, on_resume=_resume,
                              on_stop=_stop)
        # The gate hands the listener what it is suppressing. This is the only
        # way the mic is heard during a reply, and it is a queue put -- the
        # detector runs on the runner's thread, never on the audio callback.
        gate.listener = barge.feed
        timers.barge = det

        # Can the microphone hear the speakers? Without echo cancellation an
        # open mic hears the reply, and barge-in then triggers on the assistant
        # itself -- ADR-002 measured 3 spurious triggers from one 4.4 s reply.
        # This path has no AEC, so rather than print "use headphones" and hope,
        # play a chirp and look for it in the capture.
        if not args.no_echo_check and not args.wav:
            from bargein import echo_check

            verdict, corr = _echo_probe(sd, args.device)
            if verdict == "audible":
                print(f"\n  !! the microphone can hear your speakers "
                      f"(chirp correlation {corr:.2f}).\n"
                      f"     Barge-in has no echo cancellation on this path, so "
                      f"the assistant will interrupt itself.\n"
                      f"     Use headphones, or run avatar_live, whose browser "
                      f"mic has WebRTC echo cancellation.\n"
                      f"     Continuing anyway; --no-echo-check silences this.\n")
            elif verdict == "safe":
                print(f"  echo check: mic cannot hear the speakers "
                      f"(correlation {corr:.2f}) -- barge-in is safe here")
            else:
                print(f"\n  ?? echo check INCONCLUSIVE -- the microphone "
                      f"recorded silence during the probe.\n"
                      f"     That is not a pass: it means nothing was learned, "
                      f"not that the mic cannot hear the speakers.\n"
                      f"     Common on a Bluetooth headset whose mic has not "
                      f"engaged. If you are on open speakers, barge-in will "
                      f"trigger on the assistant.\n")

    print("loading (ASR + LLM + TTS, all MLX, one process) ...", flush=True)
    t0 = time.perf_counter()
    asr = from_pretrained(args.model)
    print(f"  ASR load:     {time.perf_counter() - t0:6.2f} s")

    t0 = time.perf_counter()
    chunk_samples = int(SAMPLE_RATE * args.asr_chunk_ms / 1000)
    warm = asr.create_stream(chunk_ms=args.asr_chunk_ms)
    warm.push(mx.zeros(chunk_samples))
    mx.eval()
    print(f"  ASR warm-up:  {time.perf_counter() - t0:6.2f} s  (Metal compile)")
    del warm

    max_tokens = args.max_tokens if args.max_tokens else (
        256 if args.mode == "assist" else 96)
    corrector = Corrector(args.llm, max_tokens, args.temp,
                          use_cache=not args.no_prompt_cache,
                          mode=args.mode, history_turns=args.history_turns)
    print(f"  LLM load:     {corrector.load_s:6.2f} s  ({args.llm}) "
          f"mode={args.mode}, max_tokens={max_tokens}"
          + (f", history {args.history_turns} turns" if args.mode == "assist"
             and args.history_turns else ""))
    t0 = time.perf_counter()
    if args.mode == "assist":
        corrector.respond("warm up", Turn())
        corrector.reset_history()
    else:
        corrector.correct("this are a warmup sentence", Turn())
    print(f"  LLM warm-up:  {time.perf_counter() - t0:6.2f} s"
          f"{'' if args.no_prompt_cache else '  (system prompt now cached)'}")

    engine = open_engine(args.engine, voice=args.voice)
    print(f"  TTS load:     {engine.load_s:6.2f} s + {engine.warm_s:.2f} s warm  "
          f"({engine.name}, sr {engine.sr})")
    if args.engine == "routed":
        print("                CJK engine loads on first zh/ja/ko turn "
              "(4.9 GB; skipped entirely otherwise)")

    vad = TenVad(VAD_HOP, args.threshold)
    seg = Segmenter(vad, timers, hangover_ms=args.hangover_ms,
                    min_utterance_ms=args.min_utterance_ms,
                    preroll_ms=args.preroll_ms, asr_chunk_ms=args.asr_chunk_ms)
    session = asr.create_stream(chunk_ms=args.asr_chunk_ms)

    stream = None
    mic_q: queue.Queue[np.ndarray] | None = None
    if args.wav:
        frame_source = wav_frames(args.wav, args.hangover_ms)
        print(f"\nsource: {args.wav} (offline replay through the live path)")
    else:
        q: queue.Queue[np.ndarray] = queue.Queue()
        mic_q = q

        def callback(indata, frames, time_info, status):  # audio thread only
            if status:
                timers.overruns += 1
            frame = indata[:, 0]
            # Passed so a CLOSED gate can forward it to the barge-in listener
            # rather than dropping it -- ADR-002's stated upgrade path. With
            # barge-in off there is no listener and the behaviour is unchanged.
            if not gate.accept(frame):
                return
            q.put(frame.copy())

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                blocksize=VAD_HOP, device=args.device,
                                callback=callback)

        def mic_frames():
            while True:
                try:
                    f = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                timers.max_backlog = max(timers.max_backlog, q.qsize())
                yield f

        frame_source = mic_frames()

    player = (StreamingPlayer(engine.sr, gain=args.tts_gain)
              if args.player != "none" else None)

    print(f"\nVAD hop {VAD_HOP / SAMPLE_RATE * 1000:.0f} ms, threshold {args.threshold}, "
          f"hangover {args.hangover_ms} ms, ASR chunk {args.asr_chunk_ms} ms, "
          f"lang {args.lang}")
    print("speak, then pause. Ctrl-C to stop.\n")

    turn = Turn()
    n_pushes = n_turn = 0
    if stream is not None:
        stream.start()
    try:
        for frame in frame_source:
            for kind, payload in seg.push_frame(frame):
                if kind == "start":
                    turn = Turn()
                    n_pushes = 0
                    utt_parts = []
                    print("  [speech] ", end="", flush=True)

                elif kind == "chunk":
                    # Kept for enrolment. The segmenter's chunks are the same
                    # float32 the ASR sees, so this costs a reference per chunk
                    # and no extra conversion.
                    if vprint is not None:
                        utt_parts.append(payload)
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

                    utt_audio = (np.concatenate(utt_parts)
                                 if vprint is not None and utt_parts else None)
                    if utt_audio is not None:
                        # Whoever speaks first owns barge-in. Enrolment keeps
                        # accumulating from every accepted turn afterwards --
                        # these are seconds long, where the embedding is solid,
                        # unlike the ~300 ms fragments the gate is later asked
                        # to judge.
                        emb = embedder.embed(utt_audio)
                        if emb is not None:
                            first = not vprint.enrolled
                            vprint.add(emb)
                            if first:
                                print(f"       voiceprint enrolled from "
                                      f"{len(utt_audio)/SAMPLE_RATE:.1f}s "
                                      f"({embedder.name}) -- only this voice "
                                      f"may interrupt")

                    # Half-duplex (ADR-002). The finally is load-bearing: without
                    # it a TTS failure leaves the mic deaf for the session. Note
                    # the gate reopens after player.wait() inside speak() -- i.e.
                    # once the queue has actually drained, not when synthesis
                    # returned, or our own tail lands back in the mic.
                    if stream is not None:
                        gate.close()
                    if barge is not None:
                        barge.arm()
                    try:
                        if player is not None:
                            player.reset()
                        if args.mode == "assist":
                            # Hand speak() the generator, not a string: TTS
                            # starts on sentence one while the LLM is still
                            # producing the rest.
                            said = speak(engine, player,
                                         corrector.respond_stream(raw, turn),
                                         turn, args.lang, args.voice,
                                         args.instruct, barge)
                            turn.truncated = corrector.truncated
                        else:
                            fixed = corrector.correct(raw, turn)
                            said = speak(engine, player, [fixed], turn,
                                         args.lang, args.voice, args.instruct,
                                         barge)
                        print(f"  LLM  [{n_turn}]: {said}"
                              + ("   [TRUNCATED at max-tokens]"
                                 if turn.truncated else ""))
                    finally:
                        if barge is not None:
                            barge.disarm()
                            if player is not None:
                                player.duck(1.0)
                        if stream is not None:
                            gate.reopen(mic_q, seg, args.tts_tail_ms / 1000.0)

                    timers.turns.append(turn)
                    print(f"  ---- turn {n_turn}: speech {turn.speech:.2f}s | "
                          f"asr {turn.asr_push * 1e3:.0f}ms | "
                          f"llm {turn.llm_total * 1e3:.0f}ms "
                          f"(ttft {turn.llm_ttft * 1e3:.0f}ms, "
                          f"{turn.llm_prompt_tokens} prompt tok) | "
                          f"tts ttfa {turn.tts_ttfa * 1e3:.0f}ms "
                          f"(RTF {turn.rtf:.2f}x)")
                    print(f"       END-OF-SPEECH -> AUDIO OUT: "
                          f"{turn.response * 1e3:.0f} ms\n")

    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        if player is not None:
            player.close()
        print(timers.report(args.hangover_ms, engine.name))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--wav", default=None,
                   help="replay a 16 kHz mono PCM16 WAV instead of the mic")

    p.add_argument("--model", default="dboris/nemotron-asr-mlx")
    p.add_argument("--llm", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    p.add_argument("--mode", default="assist", choices=("assist", "correct"),
                   help="'assist' answers what you say; 'correct' only fixes "
                        "the transcript's grammar and never answers (the "
                        "original behaviour)")
    p.add_argument("--history-turns", type=int, default=6,
                   help="exchanges the assistant remembers; 0 = stateless")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="0 = per-mode default (assist 256, correct 96)")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="re-evaluate the system prompt every turn (measured "
                        "+476 ms TTFT); exists as the A/B control")

    p.add_argument("--engine", default="routed",
                   help="'routed' (default: pocket-tts for en/es/fr/de/it/pt, "
                        "Qwen3-TTS lazily for zh/ja/ko), a shorthand from "
                        f"{sorted(KNOWN)}, or an HF repo id")
    p.add_argument("--voice", default="alba")
    p.add_argument("--instruct", default=None,
                   help="emotion/style instruction (Qwen3-TTS CustomVoice only)")
    p.add_argument("--lang", default="en")

    p.add_argument("--player", default="stream", choices=("stream", "none"))
    p.add_argument("--tts-gain", type=float, default=1.6,
                   help="output gain; pocket-tts peaks near 0.53 so 1.6 uses "
                        "the headroom without clipping (hard-clipped at 1.0)")
    g = p.add_argument_group("barge-in (ADR-010)")
    g.add_argument("--barge-in", action="store_true",
                   help="let a stop word interrupt a reply. Opens the mic "
                        "during playback, so WITHOUT echo cancellation it needs "
                        "headphones -- this path has none and self-checks at "
                        "startup. ADR-002's half-duplex behaviour is unchanged "
                        "when this is off")
    g.add_argument("--speaker-model", default="ecapa",
                   help="voiceprint model. 'ecapa' measured 0.0%% EER at 1.0 s "
                        "and 28.6%% at 0.3 s; 'campp' failed to separate "
                        "speakers at all (57-71%%) -- see calibrate_bargein.py")
    g.add_argument("--barge-threshold", type=float, default=0.25,
                   help="cosine a barge-in must reach against the enrolled "
                        "voiceprint to stop a reply. Higher ignores you more "
                        "often; lower lets other voices through")
    g.add_argument("--no-echo-check", action="store_true",
                   help="skip the startup chirp that tests whether the mic can "
                        "hear the speakers")
    g.add_argument("--duck-level", type=float, default=0.25,
                   help="playback volume while someone is talking over the "
                        "reply, before the stop word is confirmed")
    p.add_argument("--no-mic-gate", action="store_true")
    p.add_argument("--tts-tail-ms", type=int, default=250)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hangover-ms", type=int, default=600,
                   help="trailing silence that ends an utterance. Pure additive "
                        "latency ahead of every reply (default 600, was 800). "
                        "Measured, not guessed: 450 ms fragments -- it split "
                        "clip40 into 8 utterances including junk ('either', "
                        "'you') and degraded the transcripts; 800 ms lost the "
                        "opening phrase. 600 ms was best on both test clips")
    p.add_argument("--min-utterance-ms", type=int, default=300)
    p.add_argument("--preroll-ms", type=int, default=128)
    p.add_argument("--asr-chunk-ms", type=int, default=560)
    a = p.parse_args()
    if a.asr_chunk_ms % 80:
        p.error("--asr-chunk-ms must be a multiple of 80")
    if a.device is not None and str(a.device).isdigit():
        a.device = int(a.device)
    return a


if __name__ == "__main__":
    _a = parse_args()
    if _a.list_devices:
        list_devices()
        sys.exit(0)
    sys.exit(main(_a))
