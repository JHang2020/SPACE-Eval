# SPACE-Eval

Evaluation code for the **SPACE benchmark** from *"Human-Centric Image Captioning with Subject-Centered Spatial Understanding"*.

SPACE evaluates how well vision-language models describe **human subjects** in images, with a focus on **spatial & orientation details** (left/right, pose, action, appearance, multi-person interaction).

The evaluation is caption-based and fully automatic:

1. **Caption inference** — a VLM generates a caption for each image.
2. **LLM judge** — for every (question, ground-truth answer) pair of that image, a text-only judge model classifies whether the caption covers the information as `Correct`, `Incorrect`, or `Not Mentioned`.

## Data

The benchmark data (2,981 images, 19,892 QA pairs across 16 categories) is hosted on HuggingFace with **gated access**:

> [huggingface.co/datasets/zooblastlbz/Bench-SPACE](https://huggingface.co/datasets/zooblastlbz/Bench-SPACE) (accept the terms before downloading)

After access is granted, download and place it under `anno_data/`:

```bash
huggingface-cli download zooblastlbz/Bench-SPACE --repo-type dataset --local-dir anno_data
```

Expected layout:

```
SPACE-Eval/
├── anno_data/
│   ├── SPACE.jsonl        # QA annotations (image_path is relative to this file)
│   └── images/            # 2,981 images
└── eval/
    ├── caption_inference/ # step 1: generate captions
    └── judge/             # step 2: LLM-as-judge scoring
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1: Caption inference

```bash
cd eval/caption_inference
bash inference.sh <model_name> <model_path>
# e.g.
bash inference.sh qwen3vl-8b-thinking /path/to/Qwen3-VL-8B-Thinking
```

Supported `model_name` values (see `MODEL_REGISTRY` in `run_caption_inference.py`):

| model_name | family |
|---|---|
| `qwen3vl-8b` | Qwen3-VL-8B-Instruct |
| `qwen3vl-8b-thinking` | Qwen3-VL-8B-Thinking |
| `internvl35-8b` / `internvl35-14b` | InternVL3.5 |
| `minicpm-o45-9b` | MiniCPM-o 4.5 |
| `ovis25-9b` | Ovis2.5 |

Captions are written to `caption_results/<model_name>.jsonl`. The script is resumable: rerun the same command to retry failed images; already-finished images are skipped and existing results are never overwritten.

## Step 2: LLM judge

The judge is a text-only LLM (default: Gemma-3-27B-IT) served with vLLM:

```bash
python eval/judge/judge.py --judge_model /path/to/gemma-3-27b-it
```

By default it evaluates every `caption_results/*.jsonl` file; use `--caption_file` to evaluate a single one. Per-QA judgments are written to `eval_results/<model_name>.jsonl` with an `eval_category` field. The script checkpoints per batch and is resumable; failed or unparseable judgments are retried on the next run.

## Notes

- Both scripts resolve default paths relative to the repository root, so they can be launched from any working directory.
- Multi-GPU: step 1 uses `torchrun` data parallelism (one model replica per GPU); step 2 uses tensor parallelism across 8 GPUs by default (`TENSOR_PARALLEL_SIZE` in `judge.py`).

## License & citation

Code is released under the license in this repository. The dataset is for non-commercial research use only; see the dataset card for terms.
