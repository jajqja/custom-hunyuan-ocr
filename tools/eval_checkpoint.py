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
import json
import os
import sys
import time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processor", default=None, help="mặc định: giống --model")
    ap.add_argument("--data", required=True, help="JSONL dạng --flat")
    ap.add_argument("--out", default=None, help="ghi từng mẫu ra JSONL để soi mắt")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chấm N mẫu đầu")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    rows = [json.loads(x) for x in open(args.data, encoding="utf-8") if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} mẫu | model {args.model}")

    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).eval().cuda()

    fh = open(args.out, "w", encoding="utf-8") if args.out else None
    stats, empty_ok, empty_n = {}, 0, 0
    t0 = time.time()

    for i, r in enumerate(rows, 1):
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
        hyp = processor.decode(gen, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False).strip("\n")
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
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

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
