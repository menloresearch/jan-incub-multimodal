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

Rows list ASR services; columns list languages.
Values are WER for DE/EN/ES/FR/IT/PL/PT/RU.
CER is reported for JA/ZH-CN.
Bold indicates the lowest error rate per language.

<!-- markdownlint-disable MD013 -->
| Service                        | de        | en        | es        | fr        | it        | ja            | pl        | pt        | ru        | zh-CN         |
| ---                            | ---       | ---       | ---       | ---       | ---       | ---           | ---       | ---       | ---       | ---           |
| elevenlabs_scribe_v1           | **3.83%** | **4.56%** | **1.73%** | **5.06%** | **1.59%** | 13.34%        | 2.29%     | **3.09%** | **2.59%** | **6.13%**     |
| gladia                         | 6.79%     | 12.39%    | 3.76%     | 12.64%    | 5.67%     | 16.19%        | 7.28%     | 6.65%     | 4.69%     | 17.71%        |
| groq_whisper-large-v3          | 6.35%     | 8.33%     | 3.87%     | 11.43%    | 5.57%     | 14.78%        | 6.33%     | 5.10%     | 3.95%     | 15.01%        |
| groq_whisper-large-v3-turbo    | 7.78%     | 10.01%    | 5.80%     | 12.54%    | 5.17%     | **12.94%**    | 6.20%     | 6.34%     | 4.94%     | 16.30%        |
| fasterwhisper_whisper-large-v3 | 6.35%     | 8.52%     | 3.66%     | 11.83%    | 5.87%     | 13.89%        | 5.39%     | 4.95%     | 4.57%     | 13.42%        |
| nvidia_parakeet-tdt-0.6b-v3    | 7.78%     | 7.83%     | 2.64%     | 5.36%     | 3.28%     | not supported | **0.94%** | 7.26%     | 5.31%     | not supported |
| openai_gpt-4o-mini-transcribe  | 7.34%     | 11.10%    | 6.51%     | 11.73%    | 3.98%     | 16.68%        | 7.68%     | 8.04%     | 5.43%     | 16.48%        |
| openai_gpt-4o-transcribe       | 4.38%     | 8.62%     | 4.78%     | 9.30%     | 3.18%     | 13.46%        | 5.26%     | 5.41%     | 3.46%     | 15.07%        |
| openai_whisper-1               | 7.12%     | 7.93%     | 5.39%     | 11.73%    | 6.87%     | 15.15%        | 6.06%     | 5.41%     | 4.57%     | 29.35%        |
| speechmatics                   | 6.24%     | 9.02%     | 2.95%     | 9.71%     | 3.18%     | 16.01%        | 3.10%     | 4.79%     | 6.42%     | 17.10%        |
| vllm_openai-whisper-large-v3   | 6.57%     | 8.03%     | 3.97%     | 11.43%    | 5.77%     | 14.95%        | 5.80%     | 5.10%     | 3.70%     | 15.26%        |
<!-- markdownlint-enable MD013 -->

Average per-sample inference time (seconds).

<!-- markdownlint-disable MD013 -->
| Service | de | en | es | fr | it | ja | pl | pt | ru | zh-CN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elevenlabs_scribe_v1 | 0.96s | 0.95s | 0.84s | 1.07s | 0.85s | 0.74s | 0.80s | 0.71s | 0.81s | 0.81s |
| gladia | 6.08s | 7.09s | 5.92s | 6.71s | 6.24s | 5.71s | 6.07s | 7.04s | 7.78s | 7.35s |
| groq_whisper-large-v3 | 0.40s | 0.37s | 0.45s | 0.74s | 0.40s | 0.89s | 0.62s | 0.66s | 0.54s | 0.77s |
| groq_whisper-large-v3-turbo | 0.34s | 0.31s | 0.36s | 1.10s | 0.32s | 0.92s | 0.57s | 0.55s | 0.50s | 1.09s |
| fasterwhisper_whisper-large-v3 | 0.48s | 0.43s | 0.46s | 0.47s | 0.49s | 0.46s | 0.49s | 0.42s | 0.49s | 0.53s |
| nvidia_parakeet-tdt-0.6b-v3 | **0.08s** | **0.08s** | **0.08s** | **0.07s** | **0.08s** | not supported | **0.08s** | **0.06s** | **0.09s** | not supported |
| openai_gpt-4o-mini-transcribe | 1.06s | 1.05s | 1.08s | 1.14s | 1.07s | 1.07s | 1.12s | 1.07s | 1.02s | 1.19s |
| openai_gpt-4o-transcribe | 1.14s | 1.14s | 1.15s | 1.10s | 1.30s | 1.18s | 1.19s | 1.16s | 1.15s | 1.21s |
| openai_whisper-1 | 1.58s | 1.61s | 1.43s | 1.42s | 1.59s | 1.39s | 1.39s | 1.28s | 1.46s | 1.40s |
| speechmatics | 19.73s | 3.02s | 3.00s | 19.24s | 19.67s | 19.11s | 20.86s | 19.03s | 19.11s | 20.46s |
| vllm_openai-whisper-large-v3 | 0.37s | 0.31s | 0.35s | 0.37s | 0.36s | **0.34s** | 0.37s | 0.33s | 0.36s | **0.40s** |
<!-- markdownlint-enable MD013 -->
