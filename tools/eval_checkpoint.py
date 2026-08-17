#!/usr/bin/env python3
"""
Sinh markdown cho tập validation rồi chấm CER/WER. Một GPU, một checkpoint.

Cấu hình sinh bám theo inference/transformers/infer_hf_8gpu.py: greedy
(do_sample=False), repetition_penalty 1.08 — để số đo so được với baseline.

    python tools/eval_checkpoint.py \\
        --model ./output/colab_run/checkpoint-300 \\
        --processor ./HunyuanOCR \\
        --data ./data/flat/validation.jsonl \\
        --out ./output/eval_ckpt300.jsonl

`--processor` là bắt buộc khi chấm một `checkpoint-*`: Trainer chỉ lưu tokenizer
vào checkpoint, còn image processor thì chỉ được ghi vào thư mục gốc sau khi
train xong (`train_hunyuan.py` gọi `processor.save_pretrained` một lần ở cuối).

Nhãn rỗng (trang trắng) không tính được CER — mẫu số bằng 0 — nên chúng được
tách ra chấm bằng exact-match.
"""
import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:                                   # C++, nhanh hơn ~100x
    from rapidfuzz.distance import Levenshtein

    def edit(a, b):
        return Levenshtein.distance(a, b)
except ImportError:

    def edit(a, b):
        if a == b:
            return 0
        if not a or not b:
            return max(len(a), len(b))
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def bucket(path):
    """Tên config từ đường dẫn .../<config>/<split>/<file>.jpg."""
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) >= 3 else "?"


def make_local_generator(args):
    """Load model tại chỗ, sinh tuần tự batch 1."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    model = (HunYuanVLForConditionalGeneration
             .from_pretrained(args.model, dtype=torch.bfloat16,
                              attn_implementation=args.attn)
             .eval().cuda())

    def generate(r):
        with Image.open(r["image"]) as raw:
            image = raw.convert("RGB")
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "image", "image": r["image"]},
                                         {"type": "text", "text": r["question"]}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=image, padding=True,
                           return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 repetition_penalty=args.repetition_penalty,
                                 use_cache=True)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return processor.decode(gen, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip("\n")

    return generate


def make_server_generator(args):
    """Gửi tới endpoint OpenAI-compatible của vLLM. Ảnh nhúng dạng data URL."""
    import requests

    url = args.server.rstrip("/") + "/chat/completions"
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

    def generate(r):
        ext = os.path.splitext(r["image"])[1].lower()
        with open(r["image"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        payload = {
            "model": args.served_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime.get(ext, 'image/jpeg')};base64,{b64}"}},
                {"type": "text", "text": r["question"]},
            ]}],
            "temperature": 0.0,
            "max_tokens": args.max_new_tokens,
            # vLLM nhận tham số sampling ngoài chuẩn OpenAI ở ngay cấp cao nhất.
            # `extra_body` là khái niệm của client OpenAI, POST JSON thô mà bọc
            # vào đó thì server bỏ qua.
            "repetition_penalty": args.repetition_penalty,
        }
        resp = requests.post(url, json=payload, timeout=1800)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip("\n")

    return generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="thư mục model (chế độ tại chỗ)")
    ap.add_argument("--processor", default=None, help="mặc định: giống --model")
    ap.add_argument("--data", required=True, help="JSONL dạng --flat")
    ap.add_argument("--out", default=None, help="ghi từng mẫu ra JSONL để soi mắt")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chấm N mẫu ĐẦU tiên")
    ap.add_argument("--sample", type=int, default=0,
                    help="chấm N mẫu rải đều cả file. Dùng cái này thay --limit khi "
                         "muốn tập con đại diện: JSONL được ghi lần lượt theo từng "
                         "config nên N mẫu đầu sẽ toàn GCN")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--server", default=None,
                    help="URL vLLM OpenAI-compatible, vd http://127.0.0.1:8000/v1 . "
                         "Có cờ này thì không load model tại chỗ, gửi request song song "
                         "— nhanh hơn nhiều lần nhờ continuous batching")
    ap.add_argument("--served-name", default="tencent/HunyuanOCR",
                    help="--served-model-name lúc vllm serve")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    if not args.server and not args.model:
        ap.error("cần --model (chạy tại chỗ) hoặc --server (qua vLLM)")

    rows = [json.loads(x) for x in open(args.data, encoding="utf-8") if x.strip()]
    if args.sample and args.sample < len(rows):
        step = len(rows) / args.sample
        rows = [rows[int(i * step)] for i in range(args.sample)]
    elif args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} mẫu | {'server ' + args.server if args.server else 'model ' + args.model}")

    if args.server:
        generate = make_server_generator(args)
    else:
        generate = make_local_generator(args)

    fh = open(args.out, "w", encoding="utf-8") if args.out else None
    stats, empty_ok, empty_n = {}, 0, 0
    t0 = time.time()

    if args.server:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            hyps = list(ex.map(generate, rows))
    else:
        hyps = []
        for i, r in enumerate(rows, 1):
            hyps.append(generate(r))
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  sinh xong {len(rows)} mẫu trong {time.time() - t0:.0f}s")

    for r, hyp in zip(rows, hyps):
        ref = r["answer"].strip("\n")

        rec = {"image": r["image"], "bucket": bucket(r["image"]), "ref": ref, "hyp": hyp}
        if not ref.strip():
            empty_n += 1
            hit = not hyp.strip()
            empty_ok += hit
            rec["empty_exact_match"] = hit
        else:
            d_c = edit(ref, hyp)
            rw, hw = ref.split(), hyp.split()
            d_w = edit(rw, hw)      # cả hai cài đặt đều nhận list
            s = stats.setdefault(rec["bucket"], {"n": 0, "ce": 0, "cn": 0, "we": 0, "wn": 0})
            s["n"] += 1
            s["ce"] += d_c
            s["cn"] += len(ref)
            s["we"] += d_w
            s["wn"] += len(rw)
            rec["cer"] = round(d_c / len(ref), 4)
        if fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if fh:
        fh.close()

    print(f"\n{'tập':14}{'n':>5}{'CER':>9}{'WER':>9}")
    tot = {"ce": 0, "cn": 0, "we": 0, "wn": 0, "n": 0}
    for name, s in sorted(stats.items()):
        for k in tot:
            tot[k] += s[k]
        print(f"{name:14}{s['n']:5}{s['ce'] / s['cn']:9.4f}{s['we'] / s['wn']:9.4f}")
    if tot["cn"]:
        print(f"{'TỔNG':14}{tot['n']:5}{tot['ce'] / tot['cn']:9.4f}"
              f"{tot['we'] / tot['wn']:9.4f}")
    if empty_n:
        print(f"\nNhãn rỗng (trang trắng): {empty_ok}/{empty_n} trả đúng chuỗi rỗng")
    if args.out:
        print(f"\nChi tiết từng mẫu: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
