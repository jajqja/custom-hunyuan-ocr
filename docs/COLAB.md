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

## 4. Cài dependency

```bash
!pip install -q -U "transformers>=4.57" accelerate deepspeed safetensors \
                   binpacking lmdb tensorboard "huggingface_hub>=0.34"
```

`flash-attn` là **bắt buộc** — `train/trainer.py:3` import
`flash_attn.flash_attn_interface.flash_attn_varlen_func` ở đầu module và
`train_hunyuan.py` truyền cứng `attn_implementation="flash_attention_2"`. Build
from source trên Colab mất 40–60 phút mỗi session, nên phải lấy wheel dựng sẵn:

```python
import subprocess, sys, torch, urllib.request

tv  = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cu  = "cu" + torch.version.cuda.split(".")[0]
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py  = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"runtime: torch {tv} / {cu} / cxx11abi{abi} / {py}")

BASE = "https://github.com/Dao-AILab/flash-attention/releases/download"
def wheel_url(v):
    return (f"{BASE}/v{v}/flash_attn-{v}+{cu}torch{tv}"
            f"cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")

whl = None
for v in ("2.8.3", "2.8.2", "2.8.1", "2.8.0", "2.7.4.post1"):
    u = wheel_url(v)
    try:
        urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=20)
        whl = u
        break
    except Exception:
        print("  không có:", u.rsplit("/", 1)[1])

assert whl, "Không có wheel khớp runtime này — chọn tay ở releases page."
print("cài:", whl)
subprocess.run(["pip", "install", "-q", whl], check=True)
```

Đoạn trên dò ngược vài bản flash-attn thay vì đóng cứng một version, vì mảnh
`cu` lấy từ `torch.version.cuda` — runtime CUDA 13 sẽ tìm `cu13`, mà không phải
bản flash-attn nào cũng phát hành wheel `cu13`. Không tìm được thì mở
https://github.com/Dao-AILab/flash-attention/releases chọn tay file khớp
`torch{tv}`, `{cu}`, `cxx11abi{abi}`, `{py}` in ra ở trên.

Lưu ý `nvidia-smi` báo "CUDA Version: 13.0" là **phiên bản driver hỗ trợ**, không
phải CUDA mà torch được build. Mảnh dùng để chọn wheel là `torch.version.cuda`.

Kiểm tra:

```python
from flash_attn.flash_attn_interface import flash_attn_varlen_func   # phải im lặng
```

## 5. Đăng nhập HF, tải model + dataset

```python
from huggingface_hub import notebook_login
notebook_login()          # token cần quyền read; dataset đang private
```

```bash
# model gốc (~2 GB), bỏ nhánh v1.0
!hf download tencent/HunyuanOCR --local-dir /content/HunyuanOCR --exclude "v1.0/*"

# dataset (~600 MB) — đổi <user>/<repo> cho đúng repo bạn đã đẩy
!hf download <user>/vietnamese-doc-ocr --repo-type dataset \
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
!python tools/makedata_to_hyocr.py \
    --root /content/dataset \
    --prompt-file /content/ocr_prompt.md \
    --out-dir /content/hyocr/data/raw \
    --data-list /content/hyocr/data/data_list.txt
```

Ra 1.007 dòng train / 120 dòng validation, `thiếu ảnh` phải là 0.

```bash
%env MODEL_PATH=/content/HunyuanOCR
!MODEL_PATH=$MODEL_PATH \
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
!MODEL_PATH=/content/HunyuanOCR \
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
!hf upload <user>/hunyuanocr-vi-sft /content/hyocr/output/colab_run . \
           --repo-type model --private
```

Thư mục output có `model.safetensors` + config + processor (`train_hunyuan.py:232`
gọi `processor.save_pretrained`), đủ để load lại bằng
`HunYuanVLForConditionalGeneration.from_pretrained`.

---

## Những chỗ hay vướng

| Triệu chứng | Nguyên nhân |
|---|---|
| `ModuleNotFoundError: flash_attn` lúc load model | wheel chưa cài hoặc sai ABI — xem bước 4 |
| `RuntimeError: Default process group has not been initialized` | chạy `python train/train_hunyuan.py` trực tiếp. Phải qua `torchrun` (script đã lo), vì `train_hunyuan.py:182` gọi `torch.distributed.get_rank()` |
| Pack ra 0 dòng | đường dẫn ảnh trong `data/raw/*.jsonl` là tuyệt đối của máy cũ — phải chạy lại converter trên Colab |
| Eval vỡ ngay batch đầu | đừng bật `--eval_strategy steps`, xem `CUSTOM_SFT.md` mục 4 |
| Training đứng hình ở bước lưu | đang ghi checkpoint thẳng vào Drive — output phải nằm ở `/content` |
