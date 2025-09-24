> [!WARNING]
> This is purely experimental code. Use at your own risk.
>
# Jan Incubator Audio

This repository is for POC of `audio` and `realtime` endpoints.

## Development Setup

Make sure `uv` is installed

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a virtual environment

```bash
uv venv --python=3.12 --managed-python
source .venv/bin/activate
```

Setup pre-commit hooks

```bash
uv pip install pre-commit
pre-commit install  # install pre-commit hooks

pre-commit  # manually run pre-commit
pre-commit run --all-files  # if you forgot to install pre-commit previously
```

## Start servers

```bash
# fill api keys in .env file
cp .env.example .env

# activate venv first
souurce .venv/bin/activate

# start menlo server
python run_serve.py

# in another terminal, start whisper server (for eval purposes)
scripts/serve_whisper.sh
```

## Audio Evaluation Pipeline

- Core modules live under `audio_eval/`:
    - `common_voice_dataset.py` handles dataset sampling (defaults to `test_100` split).
    - `asr_services.py` wraps each ASR provider.
    - `text_normalizer_utils.py` wires the multilingual normalizers.
    - `wer_evaluator.py` coordinates transcription, logging, and WER scoring (HF `evaluate` + optional `tqdm`).
- CLI entry point: `scripts/run_wer_eval.py`
- Quick service checks: `scripts/test_transcribe_vllm.py` and `scripts/test_transcribe_speechmatics.py`.

Example run (progress bars + transcript dump enabled by default):

```bash
scripts/run_wer_eval.py \
  --dataset-path "$CV22_PATH" \
  --preset smoke \
  --services menlo_large-v3 vllm_openai-whisper-large-v3 speechmatics \
  --n-samples 5
```

To resume an interrupted run, point to the previous checkpoint (and optionally reuse the
same results/log paths):

```bash
scripts/run_wer_eval.py \
  --dataset-path "$CV22_PATH" \
  --preset all \
  --checkpoint results/wer_checkpoint_all_20250923_233231.json \
  --results results/wer_results_all_20250923_233231.json \
  --log logs/wer_eval_all_20250923_233231.log
```

Outputs land in `results/`, `logs/`, and `logs/transcripts_<timestamp>.jsonl`.

## Acknowledgements

This project adapts the multilingual text normalizer from the
[Hugging Face Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard),
itself derived from the Whisper repository. We appreciate the maintainers for releasing
those utilities under the Apache 2.0 license.

We also rely on community implementations of audio models such as
[faster-whisper](https://github.com/guillaumekln/faster-whisper), [vllm](https://github.com/vllm-project/vllm) and NVIDIA’s
[NeMo](https://github.com/NVIDIA/NeMo) framework, which power parts of our backend.
