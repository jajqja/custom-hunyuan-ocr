#!/usr/bin/env python3
"""
Tìm wheel flash-attn dựng sẵn khớp với runtime hiện tại.

Wheel của flash-attn được build riêng cho từng tổ hợp (CUDA major, torch minor,
cxx11 ABI, phiên bản Python, kiến trúc). Đoán tên file rồi thử tải là cách hỏng
thường xuyên: torch mới ra trước, wheel ra sau vài tuần. Script này liệt kê
asset thật từ GitHub Releases API rồi lọc, nên không bao giờ đoán.

Không khớp thì in ra bản torch gần nhất *có* wheel kèm lệnh hạ cấp — trên Colab
đó là cách duy nhất còn lại ngoài việc build from source 40-60 phút.

    python tools/pick_flash_attn.py               # chỉ in URL
    python tools/pick_flash_attn.py --install     # in rồi pip install
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request

API = "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=100&page={}"
NAME = re.compile(
    r"flash_attn-(?P<ver>[\d.a-z]+)\+cu(?P<cu>\d+)torch(?P<torch>\d+\.\d+)"
    r"cxx11abi(?P<abi>TRUE|FALSE)-(?P<py>cp\d+)-\w+-linux_(?P<arch>\w+)\.whl"
)


def verkey(s):
    return [int(x) for x in re.findall(r"\d+", s)]


def list_wheels(pages=6):
    out = []
    for page in range(1, pages + 1):
        with urllib.request.urlopen(API.format(page), timeout=30) as fh:
            releases = json.load(fh)
        if not releases:
            break
        for rel in releases:
            for asset in rel.get("assets", []):
                m = NAME.fullmatch(asset["name"])
                if m:
                    out.append((m.groupdict(), asset["browser_download_url"]))
    return out


def runtime():
    import torch
    return {
        "torch": ".".join(torch.__version__.split("+")[0].split(".")[:2]),
        "cu": torch.version.cuda.split(".")[0],
        "abi": "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE",
        "py": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "arch": "x86_64",
    }


# Chỉ số CUDA trong tên wheel là major ("cu12"), còn index của PyTorch là
# major+minor ("cu128"). Không suy ra được từ nhau nên phải tra bảng.
INDEX = {"12": "cu128", "13": "cu130"}


def plan(cu_pref, py, arch, abi):
    """Chưa có torch: chọn bản torch mới nhất còn wheel flash-attn.

    Dùng để dựng venv sạch — cài torch trước rồi mới biết có wheel hay không là
    ngược, vì torch luôn ra trước wheel vài tuần.
    """
    same = [(d, u) for d, u in list_wheels()
            if d["py"] == py and d["arch"] == arch and d["abi"] == abi]
    if not same:
        raise SystemExit(f"Không có wheel nào cho {py}/{arch}/cxx11abi{abi}.")
    target = max({d["torch"] for d, _ in same}, key=verkey)
    cands = [(d, u) for d, u in same if d["torch"] == target]
    cands.sort(key=lambda x: (x[0]["cu"] == cu_pref, verkey(x[0]["ver"])), reverse=True)
    d, url = cands[0]
    return {"torch": f"{target}.0", "cu": d["cu"],
            "index_url": f"https://download.pytorch.org/whl/{INDEX.get(d['cu'], 'cu' + d['cu'])}",
            "flash_attn": d["ver"], "wheel": url}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="in JSON {torch, index_url, wheel} mà không cần đã có torch")
    for k in ("torch", "cu", "abi", "py", "arch"):
        ap.add_argument(f"--{k}", default=None, help="ghi đè giá trị dò được")
    args = ap.parse_args()

    if args.plan:
        print(json.dumps(plan(args.cu or "12",
                              args.py or f"cp{sys.version_info.major}{sys.version_info.minor}",
                              args.arch or "x86_64",
                              args.abi or "TRUE"), indent=2))
        return 0

    env = {"torch": None, "cu": None, "abi": None, "py": None, "arch": None}
    if any(getattr(args, k) is None for k in env):
        env = runtime()
    for k in env:
        if getattr(args, k):
            env[k] = getattr(args, k)
    print(f"runtime: torch {env['torch']} / cu{env['cu']} / "
          f"cxx11abi{env['abi']} / {env['py']} / {env['arch']}")

    wheels = list_wheels()
    same_platform = [(d, u) for d, u in wheels
                     if d["py"] == env["py"] and d["arch"] == env["arch"]
                     and d["abi"] == env["abi"]]
    exact = sorted([(d, u) for d, u in same_platform
                    if d["torch"] == env["torch"] and d["cu"] == env["cu"]],
                   key=lambda x: verkey(x[0]["ver"]), reverse=True)

    if exact:
        url = exact[0][1]
        print("khớp:", url)
        if args.install:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", url], check=True)
            print("đã cài")
        return 0

    have = sorted({d["torch"] for d, _ in same_platform}, key=verkey)
    print(f"\nKhông có wheel cho torch {env['torch']}.")
    print(f"Torch có wheel ({env['py']}, cxx11abi{env['abi']}): {have}")
    if not have:
        print("Không có gì khớp nền tảng này — phải build from source.")
        return 1

    target = have[-1]
    cands = [(d, u) for d, u in same_platform if d["torch"] == target]
    # Ưu tiên giữ nguyên CUDA major của runtime để không lệch với các gói cu đã cài.
    cands.sort(key=lambda x: (x[0]["cu"] == env["cu"], verkey(x[0]["ver"])), reverse=True)
    d, url = cands[0]
    index = INDEX.get(d["cu"])
    print(f"\nGần nhất: torch {target} + cu{d['cu']}. Hạ cấp rồi cài:\n")
    print(f'  pip install -q "torch=={target}.0" torchvision \\')
    if index:
        print(f"      --index-url https://download.pytorch.org/whl/{index}")
    else:
        print(f"      --index-url https://download.pytorch.org/whl/cu{d['cu']}x  "
              f"# tra index đúng ở https://download.pytorch.org/whl/")
    print(f"  pip install -q {url}\n")
    print("Trên Colab phải RESTART runtime sau khi hạ torch:")
    print("  import os; os.kill(os.getpid(), 9)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
