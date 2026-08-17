# Chạy SFT trên Google Colab

Bản Colab của [`CUSTOM_SFT.md`](./CUSTOM_SFT.md). Notebook tương ứng:
[`notebooks/hyocr_sft_colab.ipynb`](../notebooks/hyocr_sft_colab.ipynb) — tải lên
Colab rồi chạy từ trên xuống.

> **Runtime "A100" của Colab có thể là bản 40GB hoặc 80GB, không đoán trước
> được.** Đừng đặt cứng `PACK_LEN` — bước 1 đo VRAM rồi chọn. Cùng một tài khoản
> lần này ra A100-SXM4-80GB thì lần sau vẫn có thể ra 40GB.

---

## 0. Chọn runtime

`Runtime → Change runtime type → A100 GPU` (cần Colab Pro trở lên; runtime T4
16GB **không chạy được** bài này).

## 1. Kiểm tra GPU và chọn PACK_LEN

```python
import torch, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv"],
                     capture_output=True, text=True).stdout)
GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
PACK_LEN = 16384 if GB >= 70 else 8192 if GB >= 35 else 4096
print(f"{GB:.0f} GB -> PACK_LEN={PACK_LEN}")
```

Mỗi trang A4 200 DPI tốn phần lớn ngân sách token vào vision token, nên
`PACK_LEN=8192` trên 40GB nghĩa là khoảng 2 trang một pack. Đó là bình thường —
đừng ép lên 16384 trên 40GB, sẽ OOM giữa chừng chứ không phải ngay từ step đầu.

Được **A100-80GB** thì `PACK_LEN=16384` là mặc định của bước này, giống hệt
`sft_base_1gpu.sh` trên máy riêng. Chạy hết epoch đầu, xem `nvidia-smi` còn dư
nhiều thì pack lại ở 20480 — doc upstream nói mức đó vẫn OOM được ở 80GB nên
phải đo chứ đừng đặt thẳng.

## 2. Gắn Drive

Session Colab bị ngắt là mất sạch `/content`. Gắn Drive để giữ checkpoint:

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/hyocr_sft
```

**Đừng ghi checkpoint thẳng vào Drive.** Mỗi checkpoint của model 1B kèm optimizer
state cỡ 8–12 GB; ghi qua FUSE của Drive chậm tới mức training đứng hình. Cách
làm ở bước 7: lưu vào `/content`, đồng bộ sang Drive bằng một vòng lặp nền.

## 3. Clone fork

```bash
!git clone https://github.com/jajqja/custom-hunyuan-ocr.git /content/hyocr
%cd /content/hyocr
!git log --oneline -1
```

## 4. Tạo venv và cài dependency

Dựng venv sạch ở `/content/venv` thay vì cài đè lên Python hệ thống của Colab.
Hai lý do:

1. **flash-attn bắt buộc, và chỉ có wheel cho một số bản torch.**
   `train/trainer.py:3` import `flash_attn_varlen_func` ngay đầu module, còn
   `train_hunyuan.py` truyền cứng `attn_implementation="flash_attention_2"` —
   không có đường lùi sang eager. Wheel build riêng cho từng tổ hợp **(CUDA
   major, torch minor, cxx11 ABI, Python, arch)** và torch luôn ra trước wheel
   vài tuần, nên Colab thường chạy bản torch chưa có wheel (8/2026: Colab ở
   torch 2.11, wheel mới nhất cho torch 2.10). Build from source tốn 40–60 phút
   mỗi session.
2. Hạ torch của Python hệ thống làm hỏng nửa số gói Colab cài sẵn và bắt restart
   kernel. Trong venv thì không đụng gì tới runtime.

```python
import json, os, subprocess, sys

VENV = "/content/venv"
PY = f"{VENV}/bin/python"


def run(cmd, quiet=False):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode and not quiet:
        print("$", " ".join(cmd))
        print(r.stdout[-3000:], r.stderr[-3000:], sep="")
    return r.returncode


UV = False
if run([sys.executable, "-m", "venv", VENV], quiet=True) != 0:
    print("stdlib venv không dùng được -> chuyển sang uv")
    assert run([sys.executable, "-m", "pip", "install", "-q", "uv"]) == 0
    assert run([sys.executable, "-m", "uv", "venv", "--seed", VENV]) == 0
    UV = True
assert os.path.exists(PY), "không tạo được venv"


def pipi(*args):
    cmd = ([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", PY, *args]
           if UV else [PY, "-m", "pip", "install", "-q", *args])
    assert run(cmd) == 0, f"cài thất bại: {args[0]}"


plan = json.loads(subprocess.run(
    [sys.executable, "tools/pick_flash_attn.py", "--plan"],
    capture_output=True, text=True, check=True).stdout)
print(json.dumps(plan, indent=2))

if not UV:
    pipi("-U", "pip", "setuptools", "wheel")
pipi(f"torch=={plan['torch']}", "torchvision", "--index-url", plan["index_url"])
pipi(plan["wheel"])
pipi("transformers>=4.57", "accelerate", "deepspeed", "safetensors", "datasets",
     "sentencepiece", "tokenizers", "pillow", "opencv-python-headless", "numpy",
     "einops", "tensorboard", "tqdm", "binpacking", "lmdb", "huggingface_hub>=0.34")
```

**`python -m venv` trên Colab thường hỏng.** Debian/Ubuntu tách `ensurepip` ra
gói `python3-venv` riêng, ảnh Colab không cài, nên stdlib venv thoát với mã 1.
Cell trên bắt lỗi đó và chuyển sang `uv venv --seed` (uv tự mang pip nên không
cần `ensurepip`), đồng thời cài gói bằng `uv pip` — nhanh hơn `pip` đáng kể khi
phải kéo về ~6 GB torch.

Cách khác nếu muốn giữ stdlib venv:

```bash
!apt-get install -qq -y python3.12-venv
```

Mọi lệnh cài đều **in stdout/stderr khi lỗi** thay vì chỉ ném
`CalledProcessError` không kèm thông tin gì.

`--plan` hỏi GitHub Releases API xem bản torch mới nhất nào **còn wheel**, rồi
trả về JSON:

```json
{
  "torch": "2.10.0",
  "cu": "12",
  "index_url": "https://download.pytorch.org/whl/cu128",
  "flash_attn": "2.8.1",
  "wheel": "https://github.com/Dao-AILab/.../flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
}
```

Không đoán tên file nên không mục theo thời gian; torch có lên 2.12, 2.13 thì
script vẫn chọn đúng.

Mất 5–10 phút, tốn ~6 GB đĩa (Colab A100 có sẵn hơn 100 GB).

Kiểm tra:

```python
!$VENV/bin/python -c "import torch, flash_attn; \
print(torch.__version__, torch.version.cuda, flash_attn.__version__)"
```

`nvidia-smi` báo "CUDA Version: 13.0" là **phiên bản driver hỗ trợ**, không phải
CUDA mà torch được build — driver 580.x chạy binary cu12.8 bình thường.

**Từ đây mọi lệnh shell đều thêm `PATH=$VENV/bin:$PATH`**, để `python` và
`torchrun` trong `scripts/*.sh` trỏ vào venv chứ không vào Python hệ thống.

## 5. Đăng nhập HF, tải model + dataset

```python
from huggingface_hub import notebook_login
notebook_login()          # token cần quyền read; dataset đang private
```

```bash
# model gốc (~2 GB), bỏ nhánh v1.0
!$VENV/bin/hf download tencent/HunyuanOCR --local-dir /content/HunyuanOCR --exclude "v1.0/*"

# dataset (~600 MB) — đổi <user>/<repo> cho đúng repo bạn đã đẩy
!$VENV/bin/hf download <user>/vietnamese-doc-ocr --repo-type dataset \
             --local-dir /content/dataset
```

Cây `--local-dir` khớp y hệt `hf_upload/` nên converter dùng thẳng được:

```
/content/dataset/GCN/train/{*.jpg, metadata.jsonl}
/content/dataset/GCN/validation/...
...
```

Prompt: repo dataset **không** chứa `ocr_prompt.md`, nó nằm trong `README.md`
dưới khối ```` ```text ````. Trích ra:

```python
import re, pathlib
readme = pathlib.Path("/content/dataset/README.md").read_text(encoding="utf-8")
prompt = re.search(r"````text\n(.*?)\n````", readme, re.S).group(1).strip()
pathlib.Path("/content/ocr_prompt.md").write_text(prompt, encoding="utf-8")
print(prompt[:120], "...")
```

## 6. Convert + pack

```bash
!$VENV/bin/python tools/makedata_to_hyocr.py \
    --root /content/dataset \
    --prompt-file /content/ocr_prompt.md \
    --out-dir /content/hyocr/data/raw \
    --data-list /content/hyocr/data/data_list.txt
```

Ra 1.007 dòng train / 120 dòng validation, `thiếu ảnh` phải là 0.

```bash
!PATH=$VENV/bin:$PATH \
 MODEL_PATH=/content/HunyuanOCR \
 INPUT_LIST=/content/hyocr/data/data_list.txt \
 PACK_OUTPUT=/content/hyocr/data/packed/train_${PACK_LEN}.jsonl \
 PACK_LEN=$PACK_LEN \
 NUM_PROCESSES=2 THREADS_PER_PROCESS=4 \
 FOREGROUND=1 \
     bash scripts/pack_data.sh
```

`NUM_PROCESSES=2` chứ không phải 32: Colab chỉ có 12 vCPU và mỗi process load
một bản processor riêng. Bước này mất 10–20 phút vì phải mở từng ảnh để đếm
vision token.

Kiểm tra:

```python
import json
n = t = 0
for line in open(f"/content/hyocr/data/packed/train_{PACK_LEN}.jsonl"):
    p = json.loads(line); n += 1; t += sum(x["num_tokens"] for x in p)
print(f"{n} pack, trung bình {t/n:.0f} token/pack")
```

## 7. Train

Chạy đồng bộ Drive ở nền trước (mỗi 10 phút, chỉ giữ checkpoint mới nhất):

```python
import subprocess
subprocess.Popen(
    "while true; do rsync -a --delete /content/hyocr/output/ "
    "/content/drive/MyDrive/hyocr_sft/; sleep 600; done",
    shell=True)
```

```bash
!PATH=$VENV/bin:$PATH \
 MODEL_PATH=/content/HunyuanOCR \
 TRAIN_DATA=/content/hyocr/data/packed/train_${PACK_LEN}.jsonl \
 PACK_LEN=$PACK_LEN \
 RUN_NAME=colab_run \
 SAVE_STEPS=25 \
     bash scripts/sft_base_1gpu.sh
```

`SAVE_STEPS=25` thay vì 50 vì session Colab có thể chết bất cứ lúc nào — mất tối
đa 25 step thay vì 50.

OOM thì thử theo thứ tự này (từ 80GB hạ về 8192 trước, từ 40GB hạ về 4096):

```bash
TUNE_VISION=False ...                  # đóng băng vision tower, rẻ nhất
PACK_LEN=4096 ...                      # nhớ pack lại data cùng độ dài
DEEPSPEED=scripts/zero2.json ...       # đẩy optimizer state ra, chậm hơn
```

## 8. Chạy lại sau khi mất session

`train_hunyuan.py:221` tự resume nếu `output_dir` có `checkpoint-*`. Nên sau khi
làm lại bước 3–6, copy checkpoint từ Drive về rồi chạy lại đúng lệnh bước 7 với
**cùng `RUN_NAME`**:

```bash
!mkdir -p /content/hyocr/output
!rsync -a /content/drive/MyDrive/hyocr_sft/ /content/hyocr/output/
```

Bước 6 không cần chạy lại nếu bạn cũng đồng bộ `data/packed/` sang Drive — file
packed chỉ vài MB, đáng để giữ.

## 9. Lấy model ra

```bash
!$VENV/bin/hf upload <user>/hunyuanocr-vi-sft /content/hyocr/output/colab_run . \
           --repo-type model --private
```

Thư mục output có `model.safetensors` + config + processor (`train_hunyuan.py:232`
gọi `processor.save_pretrained`), đủ để load lại bằng
`HunYuanVLForConditionalGeneration.from_pretrained`.

---

## Những chỗ hay vướng

| Triệu chứng | Nguyên nhân |
|---|---|
| `ModuleNotFoundError: flash_attn` lúc load model | đang chạy bằng Python hệ thống chứ không phải venv — thiếu `PATH=$VENV/bin:$PATH` |
| `pick_flash_attn.py` báo không có wheel nào | nền tảng lạ (aarch64, cp313 mới) — build from source là đường duy nhất |
| `RuntimeError: Default process group has not been initialized` | chạy `python train/train_hunyuan.py` trực tiếp. Phải qua `torchrun` (script đã lo), vì `train_hunyuan.py:182` gọi `torch.distributed.get_rank()` |
| Pack ra 0 dòng | đường dẫn ảnh trong `data/raw/*.jsonl` là tuyệt đối của máy cũ — phải chạy lại converter trên Colab |
| Eval vỡ ngay batch đầu | đừng bật `--eval_strategy steps`, xem `CUSTOM_SFT.md` mục 4 |
| Training đứng hình ở bước lưu | đang ghi checkpoint thẳng vào Drive — output phải nằm ở `/content` |
