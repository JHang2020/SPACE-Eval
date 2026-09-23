#!/usr/bin/env bash
# 用法: bash inference.sh <model_name> <model_path> [extra args...]
# 模型名见 run_caption_inference.py 的 MODEL_REGISTRY，例如:
#   bash inference.sh qwen3vl-8b-thinking Models/Qwen3-VL-8B-Thinking/
#   bash inference.sh internvl35-8b Models/InternVL3_5-8B-MPO/
set -e
cd "$(dirname "$0")"

python -m torch.distributed.run --nproc_per_node=8 run_caption_inference.py \
    --model "$1" --model_path "$2" "${@:3}"
