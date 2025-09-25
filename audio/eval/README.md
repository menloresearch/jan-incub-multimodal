# Audio Evaluation

Utilities here help benchmark speech-to-text services on Common Voice 22 and
other datasets using both word and character error rate (WER/CER).

## Quick Start

```bash
source ../../.venv/bin/activate
python scripts/run_wer_eval.py --help
```

- Smoke-test a single English sample against faster-whisper:

```bash
python scripts/run_wer_eval.py \
  --dataset-path "$CV22_PATH" \
  --languages en \
  --n-samples 1 \
  --services fasterwhisper_whisper-large-v3
```

- Full sweep across the default presets:

```bash
python scripts/run_wer_eval.py \
  --dataset-path "$CV22_PATH" \
  --preset all
```

- Core logic lives in `asr_services.py`, `common_voice_dataset.py`, and
  `error_rate_evaluator.py` for computing WER/CER across services.
- Example notebooks are under `notebooks/` for exploratory analysis.
- Results, logs, and transcript dumps default to the project-level
  `results/` and `logs/` directories.
