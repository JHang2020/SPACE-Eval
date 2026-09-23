import os
import re
import glob
import json
import time
import shutil
import argparse
import traceback
from collections import OrderedDict
from tqdm import tqdm

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================

# 默认路径锚定到仓库根目录，与当前工作目录无关
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLAT_QA_JSONL  = os.path.join(REPO_ROOT, "anno_data", "SPACE.jsonl")
OUTPUT_DIR     = os.path.join(REPO_ROOT, "caption_results")
MAX_NEW_TOKENS = 2048

# ==============================================================================
#                               CAPTION PROMPT
# ==============================================================================

CAPTION_PROMPT = "Please generate an English caption describing the human subjects in the image, focusing on the **Spatial & Orientation**, **Action & Pose**, **Appearance**, **Multi-person Interaction** (if applicable).\nDo not omit spatial details like left/right."

# ==============================================================================
#                               DATA LOADING
# ==============================================================================

def load_unique_images(jsonl_path: str) -> list:
    # 相对 image_path 按标注文件所在目录解析，兼容 HF 数据集的 images/ 布局
    base_dir = os.path.dirname(os.path.abspath(jsonl_path))
    seen = OrderedDict()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                key = item["image_path"]
                if not os.path.isabs(key):
                    key = os.path.normpath(os.path.join(base_dir, key))
                if key not in seen:
                    seen[key] = {"image_path": key, "raw_id": item["raw_id"]}
            except (json.JSONDecodeError, KeyError):
                continue
    return list(seen.values())


def load_finished_ids(output_path: str) -> set:
    """只把拿到非空 caption 的条目算作已完成，错误条目下次运行自动重试。"""
    finished = set()
    if not os.path.exists(output_path):
        return finished
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("caption"):
                    finished.add(item["image_path"])
            except json.JSONDecodeError:
                pass
    return finished


def append_result(output_path: str, result: dict):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def merge_results(final_output: str, ckpt_dir: str, all_images: list):
    """合并既有最终结果与所有 rank checkpoint，成功条目优先，不覆盖丢失历史数据。"""
    best = {}

    def absorb(path):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    key = item["image_path"]
                except (json.JSONDecodeError, KeyError):
                    continue
                if key not in best or (item.get("caption") and not best[key].get("caption")):
                    best[key] = item

    absorb(final_output)
    for rf in sorted(glob.glob(os.path.join(ckpt_dir, "rank*.jsonl"))):
        absorb(rf)

    order = {img["image_path"]: i for i, img in enumerate(all_images)}
    entries = sorted(best.values(), key=lambda x: order.get(x["image_path"], len(order)))

    # 先写临时文件再原子替换，中途崩溃不会损坏已有结果
    tmp_path = final_output + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp_path, final_output)

    missing = sum(1 for img in all_images
                  if not best.get(img["image_path"], {}).get("caption"))
    return len(entries), missing


# ==============================================================================
#                               MODEL BUILDERS
# ==============================================================================

def resize_for_inference(img, max_long_edge: int = 1080):
    from PIL import Image as PILImage
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / long_edge
    return img.resize((int(w * scale), int(h * scale)), resample=PILImage.LANCZOS)


def build_qwen_vl(model_path: str, device: str):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    return model, processor


def build_internvl(model_path: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    return model, tokenizer


def build_minicpm(model_path: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer, DynamicCache
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: 0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    return model, tokenizer


def build_ovis25(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    text_tok = model.text_tokenizer
    if text_tok.pad_token_id is None:
        text_tok.pad_token_id = text_tok.eos_token_id
    return model, text_tok


# ==============================================================================
#                               INFERENCE
# ==============================================================================

def _qwen_generate(model, proc, img, device: str, thinking: bool = False) -> str:
    """Qwen-VL 系列共用的生成逻辑；thinking=True 时剥离 <think> 推理链。"""
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": CAPTION_PROMPT},
        ],
    }]
    template_kwargs = {"enable_thinking": True} if thinking else {}
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        image_inputs, video_inputs = [img], None

    inputs = proc(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]

    if thinking:
        # 保留特殊 token 才能定位 </think>；未闭合说明没产出最终答案，报错走重试
        raw = proc.batch_decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        if "</think>" not in raw:
            raise RuntimeError(f"thinking chain not closed within {MAX_NEW_TOKENS} tokens")
        result = re.sub(r"<\|[^>]*\|>", "", raw.split("</think>")[-1])
    else:
        result = proc.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    torch.cuda.empty_cache()
    return result.strip()


def run_transformers_single(model, tokenizer, image_path: str,
                            model_type: str, device: str):
    """对单张图生成 caption。返回 (caption, error)，二者恰有一个非空。"""
    from PIL import Image as PILImage

    try:
        img = resize_for_inference(PILImage.open(image_path).convert("RGB"))
    except Exception as e:
        return "", f"[IMAGE_LOAD_ERROR] {e}"

    try:
        # ── Qwen3-VL（基础 / Thinking） ──────────────────────────────────────
        if model_type in ("qwen3vl", "qwen3vl_thinking"):
            caption = _qwen_generate(
                model, tokenizer, img, device,
                thinking=(model_type == "qwen3vl_thinking"),
            )

        # ── InternVL ──────────────────────────────────────────────────────────
        elif model_type == "internvl":
            import torch
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            transform = T.Compose([
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            pixel_values = transform(img).unsqueeze(0).to(
                dtype=torch.bfloat16, device=device
            )
            caption = model.chat(
                tokenizer, pixel_values, CAPTION_PROMPT,
                generation_config={"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
            ).strip()
            torch.cuda.empty_cache()

        # ── MiniCPM ───────────────────────────────────────────────────────────
        elif model_type == "minicpm":
            import torch
            msgs = [{"role": "user", "content": [img, CAPTION_PROMPT]}]
            with torch.no_grad():
                caption = str(model.chat(
                    msgs=msgs, tokenizer=tokenizer,
                    use_tts_template=False, sampling=False,
                    use_cache=False, max_new_tokens=MAX_NEW_TOKENS,
                )).strip()
            torch.cuda.empty_cache()

        # ── Ovis2.5 ───────────────────────────────────────────────────────────
        elif model_type == "ovis25":
            import torch
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": CAPTION_PROMPT},
                ],
            }]
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages, add_generation_prompt=True,
            )
            input_ids    = input_ids.to(device)
            pixel_values = (pixel_values.to(device=device, dtype=torch.bfloat16)
                            if pixel_values is not None else None)
            grid_thws    = grid_thws.to(device) if grid_thws is not None else None
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                )
            caption = model.text_tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            ).strip()
            torch.cuda.empty_cache()

        else:
            return "", f"[UNKNOWN_MODEL_TYPE] {model_type}"

    except Exception as e:
        return "", f"[INFERENCE_ERROR] {e}\n{traceback.format_exc()[:300]}"

    if not caption:
        return "", "[EMPTY_OUTPUT]"
    return caption, ""


# ==============================================================================
#                               MODEL REGISTRY
# ==============================================================================

MODEL_REGISTRY = {
    "qwen3vl-8b":          {"builder": build_qwen_vl,  "model_type": "qwen3vl"},
    "qwen3vl-8b-thinking": {"builder": build_qwen_vl,  "model_type": "qwen3vl_thinking"},
    "internvl35-8b":       {"builder": build_internvl, "model_type": "internvl"},
    "internvl35-14b":      {"builder": build_internvl, "model_type": "internvl"},
    "minicpm-o45-9b":      {"builder": build_minicpm,  "model_type": "minicpm"},
    "ovis25-9b":           {"builder": build_ovis25,   "model_type": "ovis25"},
}

# ==============================================================================
#                               MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True,
                        help=f"可选: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input",      type=str, default=FLAT_QA_JSONL)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        print(f"[ERROR] Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY.keys())}")
        return

    cfg = MODEL_REGISTRY[args.model]
    os.makedirs(args.output_dir, exist_ok=True)
    final_output = os.path.join(args.output_dir, f"{args.model}.jsonl")

    # 检测 torchrun 多进程环境
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        import torch
        RANK       = int(os.environ["RANK"])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(LOCAL_RANK)
        device     = f"cuda:{LOCAL_RANK}"
        ckpt_dir   = os.path.join(args.output_dir, f"_ckpt_{args.model}")
        os.makedirs(ckpt_dir, exist_ok=True)
        rank_output = os.path.join(ckpt_dir, f"rank{RANK}.jsonl")
        done_flag   = os.path.join(ckpt_dir, f"rank_{RANK}_DONE.txt")
        # 清掉上次中断运行残留的完成标记，避免 rank 0 提前合并
        if os.path.exists(done_flag):
            os.remove(done_flag)
    else:
        RANK, WORLD_SIZE = 0, 1
        device      = "cuda:0"
        rank_output = final_output
        ckpt_dir    = None

    # 加载数据，按 rank 分片
    all_images = load_unique_images(args.input)

    finished_ids = load_finished_ids(final_output)
    if ckpt_dir:
        for rf in glob.glob(os.path.join(ckpt_dir, "rank*.jsonl")):
            finished_ids |= load_finished_ids(rf)

    pending  = [img for img in all_images if img["image_path"] not in finished_ids]
    my_tasks = [t for i, t in enumerate(pending) if i % WORLD_SIZE == RANK]

    if RANK == 0:
        print(f"Total unique images : {len(all_images)}")
        print(f"Already finished    : {len(finished_ids)}")
        print(f"Pending             : {len(pending)}")
        print(f"This rank ({RANK}/{WORLD_SIZE}) tasks: {len(my_tasks)}")

    if my_tasks:
        model, tokenizer = cfg["builder"](args.model_path, device)
        for item in tqdm(my_tasks, desc=f"[{args.model} rank{RANK}]", disable=(RANK != 0)):
            caption, error = run_transformers_single(
                model, tokenizer, item["image_path"], cfg["model_type"], device,
            )
            append_result(rank_output, {
                "image_path": item["image_path"],
                "raw_id":     item["raw_id"],
                "model":      args.model,
                "caption":    caption,
                "error":      error,
            })
    elif RANK == 0:
        print("Nothing to do.")

    # 多进程：rank 0 等待所有 rank 完成后合并
    if is_distributed:
        with open(done_flag, "w") as f:
            f.write(f"rank {RANK} done")

        if RANK == 0:
            print(f"\nWaiting for all {WORLD_SIZE} ranks to finish...")
            pending_ranks = set(range(WORLD_SIZE))
            last_log = time.time()
            while pending_ranks:
                pending_ranks = {r for r in pending_ranks if not os.path.exists(
                    os.path.join(ckpt_dir, f"rank_{r}_DONE.txt"))}
                if not pending_ranks:
                    break
                if time.time() - last_log > 60:
                    print(f"Still waiting for ranks: {sorted(pending_ranks)}")
                    last_log = time.time()
                time.sleep(2)

            print("Merging results...")
            total, missing = merge_results(final_output, ckpt_dir, all_images)
            print(f"Merged {total} results -> {final_output}")
            if missing:
                print(f"[WARN] {missing} images still have no caption; "
                      f"checkpoints kept, rerun the same command to fill the gap.")
            else:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
    elif RANK == 0:
        print(f"\nDone! Results saved to {rank_output}")


if __name__ == "__main__":
    main()
