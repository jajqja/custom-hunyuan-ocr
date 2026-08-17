#!/usr/bin/env python3
"""
Lọc danh sách flag cho hợp với transformers đang cài, in ra stdout.

`TrainingArguments` đổi giữa transformers 4.x và 5.x: `warmup_ratio` gộp vào
`warmup_steps` (float trong [0,1) là tỷ lệ), `logging_dir` bị bỏ hẳn. Script
train mà ghim cứng tên cũ sẽ chết ngay ở `parse_args_into_dataclasses` với
"Some specified arguments are not used by the HfArgumentParser" — trước cả khi
model được load.

Cách dùng trong shell script:

    args=$("$PYTHON" tools/filter_train_args.py $args)

Flag không được nhận sẽ bị bỏ và in cảnh báo ra stderr, không âm thầm.
"""
import dataclasses
import sys

# Tên cũ -> tên mới. Chỉ đổi khi tên cũ không còn và tên mới có.
RENAMES = {"warmup_ratio": "warmup_steps"}


def known_fields():
    sys.path.insert(0, ".")
    import transformers

    from train.argument import DataArguments, ModelArguments, TrainingArguments

    names = set()
    for cls in (ModelArguments, DataArguments, TrainingArguments,
                transformers.TrainingArguments):
        names |= {f.name for f in dataclasses.fields(cls)}
    return names


def split_pairs(argv):
    """[(name, [values...])] — flag boolean có danh sách value rỗng."""
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            out.append((None, [tok]))       # giữ nguyên, không hiểu được
            i += 1
            continue
        name, vals, i = tok[2:], [], i + 1
        while i < len(argv) and not argv[i].startswith("--"):
            vals.append(argv[i])
            i += 1
        out.append((name, vals))
    return out


def main():
    names = known_fields()
    kept, dropped, renamed = [], [], []
    for name, vals in split_pairs(sys.argv[1:]):
        if name is None:
            kept += vals
            continue
        if name not in names and RENAMES.get(name) in names:
            renamed.append(f"{name} -> {RENAMES[name]}")
            name = RENAMES[name]
        if name not in names:
            dropped.append(name)
            continue
        kept.append(f"--{name}")
        kept += vals

    if renamed:
        print(f"[filter_train_args] đổi tên: {', '.join(renamed)}", file=sys.stderr)
    if dropped:
        print(f"[filter_train_args] transformers này không nhận, đã bỏ: "
              f"{', '.join(dropped)}", file=sys.stderr)
    print(" ".join(kept))


if __name__ == "__main__":
    main()
