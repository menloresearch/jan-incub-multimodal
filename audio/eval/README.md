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
- To serve the NVIDIA Parakeet ASR model locally with an OpenAI-compatible API:

  ```bash
  uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt

  source .venv/bin/activate
  python audio/eval/scripts/serve_parakeet_openai.py \
    --host 0.0.0.0 \
    --port 8010
  ```

  The server downloads `nvidia/parakeet-tdt-0.6b-v3` on first run. Point
  `asr_services.py` at it by exporting `BASE_URL=http://localhost:8010/v1`.
- Example notebooks are under `notebooks/` for exploratory analysis.
- Results, logs, and transcript dumps default to the project-level
  `results/` and `logs/` directories.
- To recompute metrics offline from a transcript dump, run
  `python scripts/score_transcripts.py path/to/transcripts.jsonl`.
  References and predictions are re-normalized per language before scoring.
  A summary is printed and saved alongside the input as
  `path/to/transcripts_summary.txt` (override with `--output`).
