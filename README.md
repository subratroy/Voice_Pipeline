# voice_pipeline

Live mic → VAD → streaming ASR → LLM → streaming TTS → speakers, all in one
process. Speak, pause, hear a reply, plus a per-turn latency breakdown printed
to the terminal when you stop it.

## Hardware requirement — read first

This **only runs on an Apple Silicon Mac** (M1/M2/M3/M4), on macOS, 16 GB RAM
recommended. Every model (ASR, LLM, TTS) runs via
[MLX](https://github.com/ml-explore/mlx), which requires Metal — there is no
Linux, Windows, or Intel-Mac build. If your machine isn't an Apple Silicon Mac,
this cannot run as-is.

## What's in this folder

```
voice_pipeline/
├── requirements.txt
├── scripts/
│   ├── voice_live.py      <- run this
│   ├── mic_vad_asr.py     mic capture + VAD segmenter (imported)
│   ├── tts_engines.py     streaming TTS engines (imported)
│   ├── voice_correct.py   LLM wrapper + prompts (imported)
│   ├── bargein.py         optional --barge-in support (imported)
│   └── speaker_id.py      optional --barge-in support (imported)
└── models/bargein/        ONNX models used only by --barge-in
```

The six scripts must stay together in one flat `scripts/` directory exactly as
shipped — `voice_live.py` finds the others via a relative import at runtime,
and `bargein.py`/`speaker_id.py` locate `models/bargein/` relative to the
`scripts/` folder's parent. Keep the whole `voice_pipeline/` folder structure
intact; don't move individual files out of it.

## Setup

```bash
cd voice_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ (3.12 recommended — that's what this was built and
tested against).

`sounddevice` needs PortAudio. The pip wheel usually bundles it, but if you see
`OSError: PortAudio library not found`, install it with:

```bash
brew install portaudio
```

## First run

**Nothing needs to be started beforehand** — no server, no Docker container,
no separate process. The ASR, LLM, and TTS models all load and run in this one
Python process.

The first run downloads about **3.3 GB** from Hugging Face and caches it in
`~/.cache/huggingface` (needs internet once; no offline fallback):

| Model | Purpose | Size |
|---|---|---|
| `dboris/nemotron-asr-mlx` | streaming speech-to-text | 2.3 GB |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | the reply/correction LLM | 839 MB |
| `mlx-community/pocket-tts-8bit` | text-to-speech (en/es/fr/de/it/pt) | 134 MB |

First launch is also slow because of model loading plus a one-time Metal
kernel compile (a few seconds) — that's a startup cost, not per-turn latency.

macOS will prompt for microphone access the first time you run it — grant it
to your terminal app (System Settings → Privacy & Security → Microphone).

## Usage

Run from inside the `voice_pipeline` folder:

```bash
python scripts/voice_live.py                        # live mic, assistant mode
python scripts/voice_live.py --list-devices          # list input devices
python scripts/voice_live.py --mode correct          # fix grammar, don't answer
python scripts/voice_live.py --engine pocket --voice marius
python scripts/voice_live.py --wav clip.wav --player none   # 16 kHz mono PCM16 only
python scripts/voice_live.py --barge-in              # headphones required, see below
```

Speak, then pause — you'll see the transcript, the reply, and a per-turn
latency line. Ctrl-C stops it and prints the full latency report.

### Key flags

| Flag | Default | What it does |
|---|---|---|
| `--llm` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | any MLX-compatible chat model repo id |
| `--voice` | `alba` | TTS voice: alba, marius, javert, jean, fantine, cosette, eponine, azelma |
| `--lang` | `en` | also es, fr, de, it, pt; zh/ja/ko lazily load a larger (4.9 GB) TTS engine on first use |
| `--mode` | `assist` | `assist` answers what you say; `correct` only fixes grammar and never answers |
| `--history-turns` | `6` | conversation turns the assistant remembers (assist mode); `0` = stateless |
| `--max-tokens` | per-mode (256 assist / 96 correct) | cap on reply length |
| `--hangover-ms` | `600` | silence needed to end an utterance — the biggest lever on felt latency |
| `--tts-gain` | `1.6` | output volume |
| `--device` | system default | input device index or name (see `--list-devices`) |

### Barge-in (`--barge-in`)

Lets you interrupt a reply by speaking a stop word (NO, STOP, WAIT, CANCEL,
SCRATCH THAT, NEVER MIND, HOLD ON, FORGET IT). Important caveats:

- **This path has no echo cancellation.** Use headphones. On open speakers,
  the microphone hears the assistant's own voice and it can interrupt itself.
  The script plays a short chirp at startup to test for this and prints a
  warning if it detects the mic can hear the speakers.
- The **first accepted utterance enrolls a voiceprint**, and only that voice
  can trigger a stop afterward.
- Uses the ONNX models bundled in `models/bargein/` — nothing extra to
  download for this feature.

## Reading the latency report

On Ctrl-C you'll see a breakdown per stage (VAD, ASR, LLM, TTS). The number
that matters most is **`END-OF-SPEECH -> AUDIO OUT`** — that's the delay the
user actually feels between finishing a sentence and hearing the reply start.
The VAD hangover (`--hangover-ms`, default 600 ms) sits in front of that and is
pure added latency by design — it's the silence the pipeline waits through
before deciding you're done talking.

## Troubleshooting

- **Wrong microphone picked up** → run `--list-devices` and pass the index or
  name via `--device`.
- **First reply in Chinese/Japanese/Korean is slow** → that's a one-time load
  of a separate ~4.9 GB TTS engine for those languages; it's skipped entirely
  if you never use them.
- **Machine feels sluggish / swapping** → this is tuned for 16 GB of memory;
  closing other heavy apps helps, especially if you also trigger the CJK TTS
  engine or barge-in's speaker model.
- **`--speaker-model campp` or `resnet34`** → not shipped in this folder
  (measurements showed `campp` performs worse than chance; `resnet34` was
  never fetched). Stick with the default `ecapa`.
