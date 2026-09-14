#!/usr/bin/env python
"""Swappable streaming TTS engines, plus a benchmark to choose between them.

Why an interface rather than one hard-coded engine: the TTS stage was 89% of the
response latency (Chatterbox v3, 9519 ms median), and picking its replacement is
a measurement problem, not a reading-the-docs problem.  Engines are therefore
comparable on identical text, and the pipeline depends on the Protocol below
rather than on any one model.

The important shape is `stream()` yielding float32 chunks instead of returning a
WAV path.  The old ChatterboxSpeaker wrote a temp file and shelled out to
`afplay`, which *structurally* cannot start playing before synthesis finishes --
time-to-first-audio was the entire synthesis, and it grew with reply length.
Chunks let playback start on the first one.

Measured on this M1 (see docs/PLAN.md ADR-003):

    pocket-tts-8bit     RTF 0.09x   TTFA ~196 ms    0.31 GB peak
    chatterbox v3       RTF 6.80x   TTFA ~9519 ms   torch, separate venv

**RTF < 1 is a hard prerequisite, not a nice-to-have.**  Above 1.0 the output
device drains faster than chunks arrive and playback stutters, no matter how
good the first-chunk latency looks in isolation.

Usage:
    .venv/bin/python scripts/tts_engines.py --bench
    .venv/bin/python scripts/tts_engines.py --bench --langs all
    .venv/bin/python scripts/tts_engines.py --say "hello there" --voice alba --play
"""

from __future__ import annotations

import argparse
import time
from typing import Iterator, Protocol

import numpy as np

# The nine languages this product must ship, per the requirements.
BENCH_TEXTS: list[tuple[str, str]] = [
    ("en", "The meeting has been moved to three o'clock tomorrow afternoon."),
    ("es", "La reunion se ha trasladado a las tres de la tarde de manana."),
    ("fr", "La reunion a ete deplacee a quinze heures demain apres-midi."),
    ("de", "Das Treffen wurde auf morgen Nachmittag um drei Uhr verschoben."),
    ("it", "La riunione e stata spostata alle tre del pomeriggio di domani."),
    ("pt", "A reuniao foi adiada para as tres horas da tarde de amanha."),
    ("zh", "会议已改到明天下午三点。"),
    ("ja", "会議は明日の午後三時に変更されました。"),
    ("ko", "회의가 내일 오후 세 시로 변경되었습니다."),
]

# Engines that have actually been measured on this machine.  Anything not in
# here is a guess until it appears in a --bench run.
KNOWN = {
    "pocket":  "mlx-community/pocket-tts-8bit",
    "pocket4": "mlx-community/pocket-tts-4bit",
    "qwen":    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "qwen06":  "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "kokoro":  "mlx-community/Kokoro-82M-bf16",
}


class TTSEngine(Protocol):
    """Everything the pipeline is allowed to assume about a TTS engine."""

    name: str
    sr: int

    def stream(self, text: str, *, lang: str = "en", voice: str | None = None,
               instruct: str | None = None) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class MLXAudioEngine:
    """mlx-audio engines: pocket-tts, Qwen3-TTS, Kokoro.

    All three run on MLX against numpy 2.5.2 -- the same runtime as the ASR and
    the LLM.  That is the whole reason to prefer this family: no torch, no
    `numpy<2` pin, no separate venv, no JSON-over-pipe worker.  Chatterbox
    needed all four.

    Two costs are paid once at construction rather than on the user's first
    turn: the Metal kernel compile, and the per-voice embedding fetch (measured
    ~5 s the first time a pocket-tts voice is used, ~0 ms thereafter).
    """

    def __init__(self, repo: str, *, voice: str | None = None,
                 warm: bool = True) -> None:
        from mlx_audio.tts.utils import load_model

        self.repo = repo
        self.name = repo.rsplit("/", 1)[-1]
        self.default_voice = voice
        t0 = time.perf_counter()
        self._model = load_model(repo)
        self.load_s = time.perf_counter() - t0
        self.sr = int(getattr(self._model, "sample_rate",
                              getattr(self._model, "sr", 24000)))

        # Warm up *through the default voice*: Qwen3-TTS CustomVoice rejects a
        # voice-less call outright, and pocket-tts pays a one-off ~5 s embedding
        # fetch per voice. Both costs belong at construction, not on the user's
        # first turn.
        self.warm_s = 0.0
        if warm:
            t0 = time.perf_counter()
            try:
                self._drain(self._gen("Warming up.", None, voice, None))
            except ValueError as e:
                raise SystemExit(
                    f"{self.name} rejected the warm-up: {e}\n"
                    "Pass --voice with a speaker this model knows."
                ) from e
            self.warm_s = time.perf_counter() - t0

    @staticmethod
    def _drain(it) -> None:
        for _ in it:
            pass

    def _gen(self, text, lang, voice, instruct, stream: bool = False):
        kw = {}
        if voice:
            kw["voice"] = voice
        if instruct:
            kw["instruct"] = instruct
        if lang:
            kw["lang_code"] = lang
        # Models disagree on which kwargs they accept; drop the optional ones
        # rather than making the caller know each model's signature.
        for attempt in (kw, {k: v for k, v in kw.items() if k != "lang_code"}, {}):
            try:
                return self._model.generate(text=text, stream=stream, **attempt)
            except TypeError:
                continue
        return self._model.generate(text=text, stream=stream)

    def stream(self, text: str, *, lang: str = "en", voice: str | None = None,
               instruct: str | None = None) -> Iterator[np.ndarray]:
        for res in self._gen(text, lang, voice or self.default_voice,
                             instruct, stream=True):
            audio = getattr(res, "audio", None)
            if audio is None:
                continue
            yield np.asarray(audio, dtype=np.float32).ravel()

    def close(self) -> None:
        self._model = None


# Language -> which engine can actually speak it.  pocket-tts is ~7x inside the
# RTF gate but has no CJK; Qwen3-TTS covers all nine but misses the gate at
# 1.38x, so it is used only where nothing faster exists.
EURO_LANGS = ("en", "es", "fr", "de", "it", "pt")
CJK_LANGS = ("zh", "ja", "ko")

# Qwen3-TTS ships native speakers per language; using an English speaker for
# Mandarin is audibly wrong, so pick per language unless told otherwise.
CJK_VOICES = {"zh": "uncle_fu", "ja": "ono_anna", "ko": "sohee"}


class RoutedEngine:
    """Two engines behind one interface, chosen per language.

    Measured (ADR-003): pocket-tts RTF 0.09-0.20x / ~260 ms TTFA, but European
    only.  Qwen3-TTS covers all nine at RTF 1.38x / ~2.7 s TTFA -- outside the
    gate, so it cannot stream smoothly, but it is still ~4x better than the
    Chatterbox baseline and it is the only licence-clean option for zh/ja/ko.

    The CJK engine is **loaded lazily**.  It peaks at 4.88 GB, and on a 16 GB
    machine already holding the ASR and the LLM, paying that for a session that
    never speaks Chinese would push the whole pipeline into swap.
    """

    def __init__(self, *, euro: str = "pocket", cjk: str = "qwen",
                 euro_voice: str | None = "alba",
                 cjk_voice: str | None = None) -> None:
        self.name = f"routed({euro}+{cjk})"
        self._euro = MLXAudioEngine(KNOWN.get(euro, euro), voice=euro_voice)
        self._cjk_spec = KNOWN.get(cjk, cjk)
        self._cjk_voice = cjk_voice
        self._cjk: MLXAudioEngine | None = None
        self.sr = self._euro.sr
        self.load_s = self._euro.load_s
        self.warm_s = self._euro.warm_s
        self.cjk_load_s = 0.0

    def _cjk_engine(self, lang: str) -> MLXAudioEngine:
        if self._cjk is None:
            t0 = time.perf_counter()
            print(f"  [loading CJK engine for '{lang}' -- first use only] ...",
                  flush=True)
            self._cjk = MLXAudioEngine(
                self._cjk_spec, voice=self._cjk_voice or CJK_VOICES.get(lang, "serena"))
            self.cjk_load_s = time.perf_counter() - t0
            if self._cjk.sr != self.sr:
                # Both are 24 kHz today; if that ever changes the player would
                # silently pitch-shift, so fail loudly instead.
                raise SystemExit(
                    f"engine sample-rate mismatch: {self._euro.name} is {self.sr} Hz "
                    f"but {self._cjk.name} is {self._cjk.sr} Hz; the shared "
                    "StreamingPlayer cannot serve both.")
        return self._cjk

    def engine_for(self, lang: str) -> tuple[MLXAudioEngine, str | None]:
        lang = (lang or "en").lower()[:2]
        if lang in CJK_LANGS:
            eng = self._cjk_engine(lang)
            return eng, self._cjk_voice or CJK_VOICES.get(lang)
        return self._euro, None   # None -> engine's own default voice

    def stream(self, text: str, *, lang: str = "en", voice: str | None = None,
               instruct: str | None = None) -> Iterator[np.ndarray]:
        eng, routed_voice = self.engine_for(lang)
        # A voice name is engine-specific ('alba' means nothing to Qwen3-TTS),
        # so an explicit --voice only applies to the engine it belongs to.
        use = voice if (voice and eng is self._euro) else routed_voice
        yield from eng.stream(text, lang=lang, voice=use, instruct=instruct)

    def close(self) -> None:
        self._euro.close()
        if self._cjk is not None:
            self._cjk.close()


def open_engine(spec: str, **kw) -> MLXAudioEngine | RoutedEngine:
    """Accept a shorthand from KNOWN, a full HF repo id, or 'routed'."""
    if spec == "routed":
        return RoutedEngine(euro_voice=kw.get("voice") or "alba")
    return MLXAudioEngine(KNOWN.get(spec, spec), **kw)


class StreamingPlayer:
    """Play float32 chunks as they arrive, via one persistent OutputStream.

    Replaces write-temp-wav-then-afplay.  The distinction that matters for the
    half-duplex gate (ADR-002) is `first_audio_at` vs `wait()`: the gate must
    stay closed until the queue has actually drained, not merely until synthesis
    returned, or the tail of our own reply lands back in the microphone.
    """

    def __init__(self, sr: int, *, blocksize: int = 1024,
                 gain: float = 1.0) -> None:
        import queue as _q

        import sounddevice as sd

        self.sr = sr
        # pocket-tts peaks around 0.53, i.e. it throws away ~5.5 dB. That is
        # audibly quiet and it also weakens the acoustic feedback path the
        # ADR-002 gate test depends on. Gain is applied with a hard clip, so
        # values above ~1.8 will distort.
        self.gain = gain
        self._q: _q.Queue[np.ndarray | None] = _q.Queue()
        self._buf = np.zeros(0, dtype=np.float32)
        self._done = False
        self.first_audio_at: float | None = None
        self.underruns = 0
        # 1.0 normally; DUCK_LEVEL while someone is talking over the reply.
        self._duck = 1.0
        self._cur_gain = gain
        self.stopped = False

        def cb(outdata, frames, time_info, status):
            need = frames
            out = np.zeros(need, dtype=np.float32)
            filled = 0
            while filled < need:
                if self._buf.size == 0:
                    try:
                        nxt = self._q.get_nowait()
                    except _q.Empty:
                        # Starved mid-utterance: the engine is slower than
                        # realtime. Audible as a gap, and the reason RTF < 1 is
                        # a hard requirement rather than a preference.
                        if not self._done and self.first_audio_at is not None:
                            self.underruns += 1
                        break
                    if nxt is None:
                        self._done = True
                        break
                    self._buf = nxt
                take = min(need - filled, self._buf.size)
                out[filled:filled + take] = self._buf[:take]
                self._buf = self._buf[take:]
                filled += take
            # Gain and duck are applied HERE, not in feed(). Barge-in has to
            # attenuate audio that is already queued -- by the time someone
            # speaks over a reply, several seconds of it are sitting in _q, and
            # scaling at feed time would leave all of that at full volume. The
            # ramp is per-block rather than instant because a step change in
            # gain is an audible click.
            g0, g1 = self._cur_gain, self.gain * self._duck
            if abs(g1 - g0) > 1e-4:
                ramp = np.linspace(g0, g1, need, dtype=np.float32)
                out *= ramp
                self._cur_gain = g1
            else:
                out *= g1
            outdata[:, 0] = np.clip(out, -1.0, 1.0)

        self._stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                       blocksize=blocksize, callback=cb)
        self._stream.start()

    def reset(self) -> None:
        """Ready the player for another utterance.

        The stream stays open across turns -- reopening an OutputStream per turn
        costs device setup and can glitch -- so the per-utterance state (done
        flag, first-audio mark, underrun count) has to be cleared explicitly.
        """
        import queue as _q

        while True:
            try:
                self._q.get_nowait()
            except _q.Empty:
                break
        self._buf = np.zeros(0, dtype=np.float32)
        self._done = False
        self.first_audio_at = None
        self.underruns = 0
        self._duck = 1.0
        self.stopped = False

    def feed(self, chunk: np.ndarray) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()
        self._q.put(np.asarray(chunk, dtype=np.float32))

    def duck(self, level: float = 1.0) -> None:
        """Scale playback without discarding it. 1.0 restores full volume.

        Used by barge-in: the dip lands ~40 ms after speech onset, long before
        anything has been recognised, and is undone by `duck(1.0)` if the
        confirmation never comes. A false trigger costs a brief dip rather than
        a lost reply.
        """
        self._duck = max(0.0, min(1.0, level))

    def stop(self) -> None:
        """Abandon the rest of this reply immediately.

        Everything queued is dropped and `wait()` stops blocking, so the caller's
        capture gate reopens now rather than after audio nobody is listening to.
        The stream itself stays open -- reopening an OutputStream per turn costs
        device setup and can glitch.
        """
        import queue as _q

        self.stopped = True
        while True:
            try:
                self._q.get_nowait()
            except _q.Empty:
                break
        self._buf = np.zeros(0, dtype=np.float32)
        self._done = True
        self._duck = 1.0

    def finish(self) -> None:
        self._q.put(None)

    def wait(self, timeout: float = 60.0) -> None:
        """Block until every queued sample has actually been played out."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self._done and self._q.empty() and self._buf.size == 0:
                break
            time.sleep(0.01)
        time.sleep(0.05)  # let the device flush its own buffer

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


def bench(engine: MLXAudioEngine, texts, *, voice=None, instruct=None) -> list[dict]:
    import mlx.core as mx

    print(f"\n{'=' * 82}\n{engine.name}\n"
          f"  load {engine.load_s:.1f}s | warm-up {engine.warm_s:.1f}s | sr {engine.sr}")
    rows = []
    for lang, text in texts:
        mx.clear_cache()
        t0 = time.perf_counter()
        ttfa, nsamp, chunks = None, 0, 0
        try:
            for ch in engine.stream(text, lang=lang, voice=voice, instruct=instruct):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                nsamp += ch.size
                chunks += 1
        except Exception as e:
            print(f"  [{lang}] FAILED {type(e).__name__}: {e}")
            continue
        total = time.perf_counter() - t0
        audio_s = nsamp / engine.sr
        rtf = total / audio_s if audio_s else float("nan")
        rows.append(dict(lang=lang, ttfa=ttfa or 0.0, synth=total, audio=audio_s,
                         rtf=rtf, chunks=chunks, peak=mx.get_peak_memory() / 1e9))
        print(f"  [{lang:<2}] ttfa {(ttfa or 0) * 1e3:6.0f}ms  synth {total * 1e3:7.0f}ms  "
              f"audio {audio_s:5.2f}s  RTF {rtf:5.2f}x  chunks {chunks:3d}")
    if rows:
        rtf = float(np.median([r["rtf"] for r in rows]))
        ttfa = float(np.median([r["ttfa"] for r in rows]))
        peak = max(r["peak"] for r in rows)
        print(f"  --> median RTF {rtf:.2f}x | median TTFA {ttfa * 1e3:.0f}ms | "
              f"peak {peak:.2f}GB | GATE RTF<0.7: "
              f"{'PASS' if rtf < 0.7 else 'FAIL'}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", default="pocket",
                   help=f"shorthand {sorted(KNOWN)} or a full HF repo id")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--langs", default="en,es,fr,de,it,pt",
                   help="comma list, or 'all' for all nine")
    p.add_argument("--say", default=None)
    p.add_argument("--voice", default=None)
    p.add_argument("--instruct", default=None,
                   help="emotion/style instruction (Qwen3-TTS CustomVoice only)")
    p.add_argument("--play", action="store_true", help="stream to the speakers")
    a = p.parse_args()

    eng = open_engine(a.engine, voice=a.voice)
    if a.bench:
        want = [l for l, _ in BENCH_TEXTS] if a.langs == "all" else a.langs.split(",")
        bench(eng, [(l, t) for l, t in BENCH_TEXTS if l in want],
              voice=a.voice, instruct=a.instruct)
        return 0

    text = a.say or "The meeting has been moved to three o'clock tomorrow afternoon."
    player = StreamingPlayer(eng.sr) if a.play else None
    t0 = time.perf_counter()
    n = 0
    for ch in eng.stream(text, voice=a.voice, instruct=a.instruct):
        if player:
            player.feed(ch)
        n += ch.size
    if player:
        ttfa = (player.first_audio_at - t0) * 1e3
        player.finish()
        player.wait()
        print(f"  ttfa {ttfa:.0f}ms | audio {n / eng.sr:.2f}s | "
              f"underruns {player.underruns}")
        player.close()
    else:
        print(f"  synth {(time.perf_counter() - t0) * 1e3:.0f}ms for {n / eng.sr:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
