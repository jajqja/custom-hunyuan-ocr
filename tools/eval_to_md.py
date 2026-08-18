#!/usr/bin/env python3
"""
Đổ kết quả `eval_checkpoint.py --out ...` thành cây .md để soi mắt.

    python tools/eval_to_md.py \\
        --eval ./eval_v2.jsonl \\
        --out ./eval_v2_md \\
        --image-root /content/hf_upload=./hf_upload

Sinh ra:

    <out>/INDEX.md                          bảng 120 mẫu, xếp theo CER giảm dần
    <out>/<config>/<tên ảnh>.md             ảnh + diff + nhãn + kết quả sinh

Mỗi file kẹp phần markdown trong fence `~~~~` chứ không phải ``` : nhãn có thể
chứa `<table>` và `|`, còn fence backtick thì vỡ nếu nội dung có backtick.

`--image-root cũ=mới` để đổi tiền tố đường dẫn: JSONL ghi đường dẫn tuyệt đối
của máy chạy eval (thường là `/content/...` trên Colab), muốn ảnh hiện được khi
mở trên máy khác thì phải trỏ lại. Link trong file là đường dẫn tương đối.
"""
import argparse
import difflib
import json
import os
import re
import sys

try:
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


FENCE = "~" * 4


def slug(name):
    """Tên file an toàn, giữ chữ có dấu (đọc được dễ hơn tên đã băm)."""
    return re.sub(r"[^\w.\-]+", "_", os.path.splitext(name)[0], flags=re.UNICODE)


def has_heading(s):
    return bool(re.search(r"^\s*#{1,6}\s", s, re.M))


def block(title, body, open_by_default=True):
    """Khối markdown thô trong <details> để file không dài mênh mông."""
    tag = "<details open>" if open_by_default else "<details>"
    return (f"{tag}\n<summary>{title}</summary>\n\n"
            f"{FENCE}markdown\n{body}\n{FENCE}\n\n</details>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="JSONL do eval_checkpoint.py --out ghi ra")
    ap.add_argument("--out", required=True, help="thư mục đích")
    ap.add_argument("--image-root", default=None, metavar="CŨ=MỚI",
                    help="đổi tiền tố đường dẫn ảnh, vd /content/hf_upload=./hf_upload")
    ap.add_argument("--label", default=None, help="tên checkpoint, hiện ở đầu INDEX.md")
    ap.add_argument("--no-images", action="store_true", help="không nhúng ảnh")
    args = ap.parse_args()

    old = new = None
    if args.image_root:
        if "=" not in args.image_root:
            ap.error("--image-root phải có dạng CŨ=MỚI")
        old, new = args.image_root.split("=", 1)
        new = os.path.abspath(new)

    with open(args.eval, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    if not rows:
        print(f"[LỖI] {args.eval} rỗng", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    out_abs = os.path.abspath(args.out)
    label = args.label or os.path.splitext(os.path.basename(args.eval))[0]

    items, missing_img = [], 0
    for r in rows:
        ref = r["ref"].strip("\n")
        hyp = r["hyp"].strip("\n")
        base = os.path.basename(r["image"])
        bucket = r.get("bucket") or "?"

        img = r["image"]
        if old and img.startswith(old):
            img = new + img[len(old):]
        if not args.no_images and not os.path.exists(img):
            missing_img += 1

        if ref.strip():
            d_c = edit(ref, hyp)
            rw, hw = ref.split(), hyp.split()
            cer = d_c / len(ref)
            wer = edit(rw, hw) / max(len(rw), 1)
        else:                       # trang trắng: CER không có mẫu số
            d_c = len(hyp)
            cer = wer = None

        items.append({
            "bucket": bucket, "base": base, "img": img, "ref": ref, "hyp": hyp,
            "cer": cer, "wer": wer, "dist": d_c, "empty_ref": not ref.strip(),
            "path": os.path.join(bucket, slug(base) + ".md"),
        })

    # Xếp theo CER giảm dần; trang trắng sai đưa lên đầu, trang trắng đúng xuống cuối.
    def rank(it):
        if it["empty_ref"]:
            return (1e9 if it["hyp"].strip() else -1.0)
        return it["cer"]

    items.sort(key=rank, reverse=True)

    for it in items:
        os.makedirs(os.path.join(out_abs, it["bucket"]), exist_ok=True)
        dst = os.path.join(out_abs, it["path"])
        lines = [f"# {it['bucket']} / {it['base']}", ""]

        if it["empty_ref"]:
            ok = not it["hyp"].strip()
            lines += [f"**Nhãn rỗng (trang trắng)** — model "
                      + ("trả đúng chuỗi rỗng ✅" if ok else
                         f"sinh ra {len(it['hyp'])} ký tự ❌"), ""]
        else:
            lines += [f"CER **{it['cer']:.4f}** · WER **{it['wer']:.4f}** · "
                      f"{it['dist']} phép sửa · nhãn {len(it['ref'])} ký tự · "
                      f"sinh {len(it['hyp'])} ký tự", ""]
            h_ref, h_hyp = has_heading(it["ref"]), has_heading(it["hyp"])
            flags = []
            if h_ref != h_hyp:
                flags.append("heading: nhãn có, sinh không" if h_ref
                             else "heading: sinh có, nhãn không")
            if ("|" in it["ref"]) != ("|" in it["hyp"]):
                flags.append("bảng markdown lệch")
            if ("<table" in it["ref"]) != ("<table" in it["hyp"]):
                flags.append("bảng HTML lệch")
            for ph in ("[Chữ ký]", "[Mã vạch]"):
                if it["ref"].count(ph) != it["hyp"].count(ph):
                    flags.append(f"`{ph}` {it['ref'].count(ph)} → {it['hyp'].count(ph)}")
            if len(it["hyp"]) > 1.4 * max(len(it["ref"]), 1):
                flags.append("phần sinh dài bất thường — nghi lặp")
            if flags:
                lines += ["> ⚠ " + " · ".join(flags), ""]

        lines += [f"[← INDEX](../INDEX.md)", ""]

        if not args.no_images:
            rel = os.path.relpath(it["img"], os.path.dirname(dst))
            lines += [f"![{it['base']}]({rel})", ""]

        if not it["empty_ref"] and it["dist"]:
            diff = list(difflib.unified_diff(
                it["ref"].split("\n"), it["hyp"].split("\n"),
                "nhãn", "sinh", lineterm="", n=1))
            lines += ["<details open>", "<summary>Khác biệt theo dòng</summary>", "",
                      "```diff", *diff, "```", "", "</details>", ""]

        lines += [block("Model sinh ra", it["hyp"] or "(rỗng)"),
                  block("Nhãn", it["ref"] or "(rỗng)", open_by_default=False)]

        with open(dst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    # ---- INDEX ----
    stats = {}
    for it in items:
        if it["empty_ref"]:
            continue
        s = stats.setdefault(it["bucket"], {"n": 0, "ce": 0, "cn": 0})
        s["n"] += 1
        s["ce"] += it["dist"]
        s["cn"] += len(it["ref"])

    idx = [f"# Kết quả OCR — {label}", "",
           f"{len(items)} mẫu · xếp theo CER giảm dần, mẫu tệ nhất ở trên.", "",
           "| tập | n | CER |", "|---|---:|---:|"]
    tot = {"n": 0, "ce": 0, "cn": 0}
    for name, s in sorted(stats.items(), key=lambda x: -x[1]["n"]):
        for k in tot:
            tot[k] += s[k]
        idx.append(f"| {name} | {s['n']} | {s['ce'] / s['cn']:.4f} |")
    if tot["cn"]:
        idx.append(f"| **TỔNG** | **{tot['n']}** | **{tot['ce'] / tot['cn']:.4f}** |")
    idx += ["", "| # | CER | WER | tập | mẫu |", "|---:|---:|---:|---|---|"]
    for i, it in enumerate(items, 1):
        link = f"[{it['base']}]({it['path'].replace(os.sep, '/')})"
        if it["empty_ref"]:
            mark = "trắng ✅" if not it["hyp"].strip() else "trắng ❌"
            idx.append(f"| {i} | {mark} | — | {it['bucket']} | {link} |")
        else:
            idx.append(f"| {i} | {it['cer']:.4f} | {it['wer']:.4f} | "
                       f"{it['bucket']} | {link} |")

    with open(os.path.join(out_abs, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(idx) + "\n")

    print(f"{len(items)} file trong {args.out}/  ·  mở {args.out}/INDEX.md")
    if missing_img:
        print(f"[cảnh báo] {missing_img}/{len(items)} ảnh không tồn tại tại đường dẫn "
              f"đã đổi — kiểm lại --image-root", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
