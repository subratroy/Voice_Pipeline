#!/usr/bin/env python
"""Stop-word barge-in: let the user interrupt a reply, but only deliberately.

    frames (16 ms) -> VAD -> keyword spotter -> voiceprint -> DUCK | STOP | RESUME

ADR-002 closed the microphone during playback because, ungated, the pipeline
transcribed its own reply and answered itself in a loop -- measured, 3 spurious
triggers from one 4.4 s reply through the speakers. This reopens it, narrowly:
the reply only stops for a stop word, spoken by the enrolled speaker.

Duck, then confirm
------------------
    t=0ms     speech detected     -> DUCK    volume 100% -> 25%
    t~300ms   keyword + speaker   -> STOP    flush the reply, reopen the mic
              anything else       -> RESUME  volume back to 100%

A false positive costs a 300 ms dip, not a lost reply. That asymmetry is what
makes it safe to listen at all.

Why "no" ducks but never stops
------------------------------
Two independent measurements condemned it, which is why it is special-cased
rather than simply given a stricter threshold:

* **The keyword spotter cannot hold it.** Swept across thresholds 0.05-0.35 on
  synthesised speech, a standalone "No." was *never* detected -- while at boost
  1.0 it fired inside "there is **no** rush". Missed when meant, fired when not,
  with no setting in between.
* **The voiceprint cannot verify it.** A spoken "no" is ~300 ms, where ECAPA
  measured **28.6% EER** against 0.0% at 1.0 s (`calibrate_bargein.py --compare`).

So "no" ducks the audio -- which is the responsive part users actually feel --
and a phrase like "scratch that" is what stops the reply. Those run ~1 s, where
both components are reliable.

The same reasoning generalises past the word list: `MIN_STOP_S` refuses a hard
stop on any segment too short to verify, whatever was said in it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from mic_vad_asr import SAMPLE_RATE, VAD_HOP  # noqa: E402

KWS_DIR = os.path.join(_ROOT, "models", "bargein",
                       "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")

# Measured (see the sweep in ADR-010): boost 2.0 with a low threshold gave 4/5
# stop words detected and 0/4 false fires on ordinary speech containing them.
# At boost 1.0 the same audio produced 3/5 and 2/4 -- worse on both axes.
KWS_BOOST = 2.0
KWS_THRESHOLD = 0.10

# Words that may only duck, never stop. See the module docstring.
DUCK_ONLY = frozenset({"NO"})

# A hard stop needs enough audio for the voiceprint to mean something. ECAPA
# measured 0.0% EER at 1.0 s, 14.3% at 0.5 s and 28.6% at 0.3 s, so this is the
# line between "verified" and "guessed". A keyword landing earlier keeps the
# audio ducked and is re-checked as more speech arrives.
MIN_STOP_S = 0.7

# How long to keep listening after a duck before giving up and resuming.
CONFIRM_WINDOW_S = 2.0

# Silence needed before a ducked window is abandoned. Not optional: "never mind"
# and "scratch that" both contain an internal pause, and resuming on the first
# silent frame tore the phrase in half -- the spotter's stream was reset
# mid-phrase and neither word was ever recognised. Measured: with no hangover,
# 2 of 5 stop words were missed that the same model detected fine when fed as a
# whole utterance. The segmenter has the same guard for the same reason.
SILENCE_HANGOVER_S = 0.35

DUCK, STOP, RESUME = "duck", "stop", "resume"


@dataclass
class Decision:
    kind: str                       # DUCK | STOP | RESUME
    keyword: str | None = None
    score: float = 0.0              # cosine against the voiceprint
    seconds: float = 0.0            # audio the decision was made on
    reason: str = ""

    def row(self) -> str:
        k = f' kw="{self.keyword}"' if self.keyword else ""
        s = f" cos={self.score:+.3f}" if self.keyword else ""
        return (f"  barge-in: {self.kind.upper():6s}{k}{s} "
                f"({self.seconds:.2f}s) {self.reason}")


@dataclass
class Stats:
    ducks: int = 0
    stops: int = 0
    resumes: int = 0
    kw_hits: int = 0
    rejected_speaker: int = 0
    rejected_short: int = 0
    rejected_duckonly: int = 0
    scores: list = field(default_factory=list)

    frames_in: int = 0

    def row(self) -> str:
        if not self.frames_in:
            return ("  barge-in:            armed but NO MICROPHONE FRAMES "
                    "arrived -- permission denied, or the page never called "
                    "getUserMedia")
        heard = f"{self.frames_in * 256 / 16000:.0f}s of mic audio"
        if not self.ducks:
            return (f"  barge-in:            armed, {heard}, never triggered "
                    f"(the VAD heard no speech in it)")
        rej = (f", rejected {self.rejected_speaker} speaker / "
               f"{self.rejected_short} too-short / "
               f"{self.rejected_duckonly} duck-only")
        cos = (f", cos median {np.median(self.scores):+.3f}"
               if self.scores else "")
        return (f"  barge-in:            {heard}, {self.ducks} ducks, "
                f"{self.stops} stops, {self.kw_hits} keyword hits{rej}{cos}")


class BargeInDetector:
    """Feed it 16 ms int16 frames while the assistant is speaking.

    Source-agnostic on purpose: the frames come from the browser's echo-cancelled
    `getUserMedia` in `avatar_live`, and straight from PortAudio in
    `voice_live.py` / `voice_avatar.py`. Only the echo handling differs, and that
    is the caller's problem -- see the headphone self-check in `echo_check()`.
    """

    def __init__(self, voiceprint, embedder, *, threshold: float = 0.35,
                 vad_threshold: float = 0.5, on_decision=None) -> None:
        import sherpa_onnx
        from ten_vad import TenVad

        if not os.path.isdir(KWS_DIR):
            raise SystemExit(
                f"keyword-spotting model missing: "
                f"{os.path.relpath(KWS_DIR, _ROOT)}\n"
                f"Fetch it with:\n"
                f"  .venv/bin/python scripts/fetch_bargein_models.py")
        kw = os.path.join(KWS_DIR, "stopwords.txt")
        if not os.path.exists(kw):
            raise SystemExit(f"stop words not tokenised: {kw}\n"
                             f"Regenerate with scripts/make_stopwords.py")
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(KWS_DIR, "tokens.txt"),
            encoder=os.path.join(KWS_DIR, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            decoder=os.path.join(KWS_DIR, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
            joiner=os.path.join(KWS_DIR, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
            keywords_file=kw, keywords_score=KWS_BOOST,
            keywords_threshold=KWS_THRESHOLD, num_threads=1, provider="cpu")
        # Its own VAD instance. ten-vad keeps internal feature history and has no
        # reset, so sharing the segmenter's would mean the barge-in listener and
        # the utterance segmenter corrupting each other's state. 336 us a frame
        # is not worth that.
        self._vad = TenVad(VAD_HOP, vad_threshold)
        self.vp = voiceprint
        self.em = embedder
        self.threshold = threshold
        self._on = on_decision or (lambda d: None)
        self.stats = Stats()
        self._reset()

    def _reset(self) -> None:
        self._active = False           # ducked, listening for a confirmation
        self._buf: list[np.ndarray] = []
        self._stream = None
        self._fired = None             # keyword seen but not yet actionable
        self._silence = 0              # consecutive non-speech frames

    # -- the hot path ---------------------------------------------------
    def push(self, frame_i16: np.ndarray) -> Decision | None:
        """One VAD-hop frame. Returns a Decision when the state changes."""
        _p, speech = self._vad.process(frame_i16)

        if not self._active:
            if not speech:
                return None
            # Duck first, ask questions after. This is the whole reason the
            # feature feels instant: the dip lands ~40 ms after speech onset,
            # long before anything has been recognised.
            self._active = True
            self._buf = [frame_i16.copy()]
            self._stream = self._kws.create_stream()
            self._feed(frame_i16)
            self.stats.ducks += 1
            d = Decision(DUCK, seconds=0.0, reason="speech detected")
            self._on(d)
            return d

        self._buf.append(frame_i16.copy())
        self._feed(frame_i16)
        secs = sum(len(b) for b in self._buf) / SAMPLE_RATE
        # Ride through the pause inside a phrase rather than ending on it.
        self._silence = 0 if speech else self._silence + 1
        quiet_s = self._silence * VAD_HOP / SAMPLE_RATE

        hit = self._poll()
        if hit:
            self.stats.kw_hits += 1
            self._fired = hit

        if self._fired:
            d = self._judge(self._fired, secs)
            if d is not None:
                self._reset()
                self._on(d)
                return d

        # Nothing confirmed in time, or the speaker really has stopped: let the
        # reply run on.
        if secs >= CONFIRM_WINDOW_S or quiet_s >= SILENCE_HANGOVER_S:
            self._reset()
            self.stats.resumes += 1
            d = Decision(RESUME, seconds=secs,
                         reason="no stop word" if not self._fired
                                else "not confirmed")
            self._on(d)
            return d
        return None

    def _feed(self, frame_i16: np.ndarray) -> None:
        self._stream.accept_waveform(
            SAMPLE_RATE, frame_i16.astype(np.float32) / 32768.0)

    def _poll(self) -> str | None:
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            r = self._kws.get_result(self._stream)
            if r:
                self._kws.reset_stream(self._stream)
                return r
        return None

    def _judge(self, keyword: str, secs: float) -> Decision | None:
        """Decide whether `keyword` may stop the reply. None = keep listening."""
        if keyword in DUCK_ONLY:
            # Not a threshold decision -- this word is unreliable in the spotter
            # AND unverifiable at its length. Stay ducked and wait for a phrase.
            self.stats.rejected_duckonly += 1
            return Decision(RESUME, keyword, seconds=secs,
                            reason="duck-only word; needs a phrase to stop")
        if secs < MIN_STOP_S:
            # Too short to verify. Not a rejection -- keep buffering and re-judge
            # as the speaker keeps talking. This is what turns a borderline
            # 300 ms hit into a solid 1 s one a beat later.
            self.stats.rejected_short += 1
            return None
        if not self.vp.enrolled:
            return Decision(STOP, keyword, seconds=secs,
                            reason="no voiceprint enrolled yet")
        audio = np.concatenate(self._buf).astype(np.float32) / 32768.0
        emb = self.em.embed(audio)
        if emb is None:
            return None
        score = self.vp.score(emb)
        self.stats.scores.append(score)
        if score >= self.threshold:
            self.stats.stops += 1
            return Decision(STOP, keyword, score, secs, "verified")
        self.stats.rejected_speaker += 1
        return Decision(RESUME, keyword, score, secs,
                        f"speaker below {self.threshold:+.2f}")


def echo_check(playrec, seconds: float = 0.6) -> tuple[str, float]:
    """Can the microphone hear the speakers?

    Returns (verdict, correlation) where verdict is "audible" (the mic hears the
    output -- barge-in will trigger on the assistant itself), "safe" (it does
    not), or "inconclusive" (the capture was silent, so nothing was learned).

    Without echo cancellation an open mic hears the assistant, and barge-in then
    triggers on the reply itself. Rather than tell the user to wear headphones
    and hope, play a chirp and look for it in the capture.

    `playrec(samples) -> recording` must play and record **simultaneously**; the
    caller supplies it so this works for both the PortAudio and browser paths.
    One function rather than separate play/record on purpose: sounddevice's
    module-level `sd.play` and `sd.rec` share a single stream, so calling `rec`
    after `play` silently cancels the playback -- the probe then measures nothing
    and reports "safe" every time, which is far worse than not checking at all.
    `sd.playrec` is the one that does both.
    """
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    # A sweep, not a tone: it correlates sharply and is easy to tell from room
    # noise that happens to sit at one frequency.
    chirp = (0.25 * np.sin(2 * np.pi * (300 + (3000 - 300) * t / seconds / 2) * t)
             ).astype(np.float32)
    got = np.asarray(playrec(chirp), dtype=np.float32).reshape(-1)
    if got.size < n:
        return "inconclusive", 0.0
    # A silent capture proves nothing. It means the microphone heard no chirp,
    # but equally it means the microphone heard *nothing* -- muted, disconnected,
    # or a Bluetooth headset whose HFP mic did not engage. Reporting "safe" from
    # that is exactly the false confidence this probe exists to prevent.
    # ADR-002's feedback test takes the same line: when the control arm shows
    # zero it reports INCONCLUSIVE rather than a pass.
    if float(np.abs(got).max()) < 1e-4:
        return "inconclusive", 0.0
    a = chirp - chirp.mean()
    b = got[:n] - got[:n].mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    corr = float(np.max(np.abs(np.correlate(b, a, mode="valid"))) / denom)
    return ("audible" if corr > 0.15 else "safe"), corr


class BargeInRunner:
    """Runs a `BargeInDetector` off the audio thread.

    The gate's `accept()` is called from CoreAudio's callback, so it may only
    enqueue -- keyword spotting and a speaker embedding there would starve the
    output stream and be heard as a dropout. This owns the queue and the worker.

    `armed` gates the whole thing: frames are only examined while the assistant
    is actually speaking. Outside that window the microphone is the segmenter's
    and this must not touch it.
    """

    def __init__(self, detector: BargeInDetector, *, on_duck=None, on_stop=None,
                 on_resume=None) -> None:
        import queue
        import threading

        self.det = detector
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._armed = False
        self._cbs = {DUCK: on_duck, STOP: on_stop, RESUME: on_resume}
        self.stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def arm(self) -> None:
        """Begin listening. Called when the assistant starts speaking."""
        import queue

        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.det._reset()
        self.stopped = False
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def feed(self, frame_i16) -> None:
        """Audio-thread safe: enqueue only, never block."""
        # Counted even when disarmed: "frames arrive but only between replies"
        # is a real fault mode and it has to be distinguishable from silence.
        self.det.stats.frames_in += 1
        if not self._armed:
            return
        try:
            self._q.put_nowait(np.asarray(frame_i16).copy())
        except Exception:
            # Full queue means the worker is behind. Dropping a frame degrades
            # detection slightly; blocking here would glitch playback outright.
            pass

    def _run(self) -> None:
        import queue

        while True:
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self._armed:
                continue
            try:
                d = self.det.push(frame)
            except Exception as e:                  # never kill the worker
                print(f"  [barge-in] {type(e).__name__}: {e}")
                continue
            if d is None:
                continue
            cb = self._cbs.get(d.kind)
            if d.kind == STOP:
                self.stopped = True
                self._armed = False
            if cb:
                cb(d)
