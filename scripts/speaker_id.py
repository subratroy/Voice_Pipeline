#!/usr/bin/env python
"""Voiceprints: enrol a speaker, then score later audio against them.

Used by the barge-in gate (`bargein.py`) to answer one question: is the person
talking over the reply the same person who started this session?

    enrol   first accepted utterance, 2-6 s  ->  embedding (512-d for CAM++,
                                                 192-d for ECAPA)
    score   a barge-in segment, ~0.3-1.5 s   ->  cosine against the centroid

Two things to be honest about before relying on this.

**Short utterances are the hard case, and barge-in is made of them.** Published
figures put EER at 8.72% for 3.59 s utterances and 12.8% at 2.05 s -- a 46%
relative increase for a 1.5 s difference. A spoken "no" is roughly 300 ms, well
below either. That degradation is a property of the task, not of any particular
model, so it is not fixed by choosing a different architecture. Three things here
reduce it: `score()` takes the whole VAD segment rather than the keyword, the
caller may re-score as more audio arrives, and `enrol()` accumulates a centroid
over the session instead of trusting one sample.

**What it is actually good at here is rejecting the assistant.** Residual echo
that survives the browser's AEC is a synthetic TTS voice, which separates from an
enrolled human far more cleanly than one human separates from another. That is
the discrimination worth leaning on; human-vs-human at 300 ms is the weak half.

No torch. `speechbrain/spkrec-ecapa-voxceleb` is the best-known model in this
space and has the best published number (0.80% EER, VoxCeleb1-O cleaned), but it
resolves to torch + torchaudio + speechbrain -- 37 packages -- and ADR-005
removed torch from this stack deliberately. Any ONNX graph with the right input
signature works here; see `calibrate_bargein.py --compare` for choosing one on
your own audio rather than on a leaderboard.
"""

from __future__ import annotations

import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models", "bargein")

# name -> (filename, note). Fetched by `voice_avatar.py --fetch-bargein-models`.
MODELS = {
    "campp": ("wespeaker_en_voxceleb_CAM++_LM.onnx",
              "wespeaker CAM++ large-margin, VoxCeleb English, official k2-fsa release"),
    "ecapa": ("ecapa_voxceleb.onnx",
              "speechbrain ECAPA-TDNN, exported locally and parity-checked "
              "(scripts/export_ecapa.py)"),
    "resnet34": ("wespeaker_en_voxceleb_resnet34_LM.onnx",
                 "wespeaker ResNet34 large-margin, VoxCeleb English"),
}
# Measured, not assumed. `calibrate_bargein.py --compare` enrolled one voice and
# scored seven others across four segment lengths:
#
#     dur    CAM++ EER    ECAPA EER
#     0.3 s     57.1%        28.6%
#     0.5 s     71.4%        14.3%
#     1.0 s     57.1%         0.0%
#     2.0 s     57.1%         0.0%
#
# CAM++ does not separate these speakers at all -- impostors scored HIGHER than
# the enrolled speaker (+0.368 vs +0.342 at 0.3 s), which is worse than chance.
# It is kept in MODELS so the comparison can be reproduced rather than taken on
# trust. ECAPA is 3x slower (16x vs 48x realtime) and worth every millisecond.
DEFAULT = "ecapa"

SAMPLE_RATE = 16000
# Below this there is not enough speech for an embedding to mean anything. The
# caller gets None rather than a confident-looking number computed from 80 ms of
# audio -- a wrong score is worse than an absent one, because the gate acts on it.
MIN_SPEECH_S = 0.25


class Voiceprint:
    """One enrolled speaker: a running centroid of L2-normalised embeddings."""

    def __init__(self) -> None:
        self.vec: np.ndarray | None = None
        self.n = 0

    def add(self, emb: np.ndarray) -> None:
        """Fold another sample in. Enrolment improves over a session.

        A plain running mean, not a weighted one: every contribution is a full
        accepted turn (seconds long), so there is no reason to trust the first
        one more than the fifth. The mean is re-normalised because cosine
        against an unnormalised centroid is not the same ordering.
        """
        e = emb / (np.linalg.norm(emb) + 1e-9)
        self.vec = e if self.vec is None else self.vec + (e - self.vec) / (self.n + 1)
        self.vec = self.vec / (np.linalg.norm(self.vec) + 1e-9)
        self.n += 1

    @property
    def enrolled(self) -> bool:
        return self.vec is not None

    def score(self, emb: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]. Higher is more likely the same speaker."""
        if self.vec is None:
            return 0.0
        e = emb / (np.linalg.norm(emb) + 1e-9)
        return float(np.dot(self.vec, e))


class SpeakerEmbedder:
    """One ONNX speaker-embedding model. Two backends, one interface.

    **sherpa-onnx** for the wespeaker and 3D-Speaker models, because each wants
    its own feature frontend -- 80-dim fbank with model-specific normalisation --
    and sherpa bundles kaldi-native-fbank configured per model. Reproducing that
    by hand is where an embedding silently becomes garbage: the graph still runs
    and still returns plausible floats.

    **Plain onnxruntime** for our ECAPA export, which does not fit sherpa's
    expected model interface. That is fine here precisely because the frontend
    was folded into the exported graph (`export_ecapa.py`), so it takes a
    waveform directly and there is no frontend left to get wrong -- the parity
    check at export time covers the whole chain rather than just the encoder.
    """

    def __init__(self, name: str = DEFAULT, *, num_threads: int = 1) -> None:
        if name not in MODELS:
            raise SystemExit(f"unknown speaker model {name!r}. "
                             f"Known: {', '.join(MODELS)}")
        fn, self.note = MODELS[name]
        path = os.path.join(MODEL_DIR, fn)
        if not os.path.exists(path):
            raise SystemExit(
                f"speaker model missing: {os.path.relpath(path, _ROOT)}\n"
                + ("Export it with:\n"
                   "  .venv-export/bin/python scripts/export_ecapa.py"
                   if name == "ecapa" else
                   "Fetch it with:\n"
                   "  .venv/bin/python scripts/fetch_bargein_models.py"))
        self.name = name
        self.calls = 0
        self.total_s = 0.0
        self.infer_s = 0.0
        # One thread: this runs alongside ASR, LLM and TTS on 8 cores, and the
        # segments are short enough that thread setup would dominate anyway.
        if name == "ecapa":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = num_threads
            opts.log_severity_level = 3
            # The 83 MB of weights live in a sibling `.onnx.data`; onnxruntime
            # resolves it relative to the model path, so the two files have to
            # stay together.
            self._sess = ort.InferenceSession(
                path, opts, providers=["CPUExecutionProvider"])
            self.dim = self._sess.get_outputs()[0].shape[-1]
            self._backend = "onnxruntime"
        else:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=path, num_threads=num_threads, provider="cpu")
            self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self.dim = self._ex.dim
            self._backend = "sherpa-onnx"

    def embed(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray | None:
        """float32 [-1,1] mono -> embedding, or None if there is too little audio."""
        import time

        a = np.asarray(audio, dtype=np.float32)
        if a.size < MIN_SPEECH_S * sr:
            return None
        t0 = time.perf_counter()
        if self._backend == "onnxruntime":
            emb = self._sess.run(None, {"wav": a.reshape(1, -1),
                                        "lens": np.ones(1, dtype=np.float32)})[0]
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        else:
            s = self._ex.create_stream()
            s.accept_waveform(sample_rate=sr, waveform=a)
            s.input_finished()
            if not self._ex.is_ready(s):
                return None
            emb = np.asarray(self._ex.compute(s), dtype=np.float32)
        self.infer_s += time.perf_counter() - t0
        self.total_s += a.size / sr
        self.calls += 1
        return emb


def main() -> int:
    """Self-test: embed two halves of one clip and check they look like one person.

        .venv/bin/python scripts/speaker_id.py [wav] [--model campp]

    A single speaker split in half should score high against itself. It is a
    sanity check on the frontend and the graph, not an accuracy measurement --
    for that, use scripts/calibrate_bargein.py.
    """
    import argparse
    import wave

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("wav", nargs="?", default=os.path.join(
        _ROOT, "samples", "avatar", "shared_audio.wav"))
    p.add_argument("--model", default=DEFAULT, choices=sorted(MODELS))
    a = p.parse_args()

    with wave.open(a.wav) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    audio = pcm.astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)

    em = SpeakerEmbedder(a.model)
    print(f"{a.model}: {em.note}\n  dim {em.dim}, {len(audio)/SAMPLE_RATE:.2f} s of audio")

    half = len(audio) // 2
    e1, e2 = em.embed(audio[:half]), em.embed(audio[half:])
    if e1 is None or e2 is None:
        print("  too short to embed"); return 1
    vp = Voiceprint(); vp.add(e1)
    same = vp.score(e2)
    noise = em.embed(np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.05)
    vs_noise = vp.score(noise) if noise is not None else float("nan")

    print(f"  same speaker, two halves : {same:+.3f}")
    print(f"  vs white noise           : {vs_noise:+.3f}")
    print(f"  {em.infer_s*1e3:.0f} ms for {em.total_s:.1f} s of audio "
          f"({em.total_s/max(em.infer_s,1e-9):.0f}x realtime)")
    ok = same > 0.5 and same - vs_noise > 0.3
    print(f"\n  {'OK' if ok else 'SUSPECT'} -- self-similarity should be high and "
          f"clearly above noise")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
