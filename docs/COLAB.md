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
import json, os, shutil, subprocess, sys

VENV = "/content/venv"
PY = f"{VENV}/bin/python"


def run(cmd, quiet=False):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode and not quiet:
        print("$", " ".join(cmd))
        print(r.stdout[-3000:], r.stderr[-3000:], sep="")
    return r.returncode


# Dùng uv chứ không phải `python -m venv`: Debian tách ensurepip ra gói
# python3-venv riêng mà ảnh Colab không cài, nên stdlib venv thoát mã 1 và để
# lại một thư mục hỏng dở. uv tự mang pip, và cài nhanh hơn hẳn khi phải kéo
# về ~6 GB torch.
assert run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"]) == 0

if not os.path.exists(PY):                    # chạy lại cell không cài lại từ đầu
    shutil.rmtree(VENV, ignore_errors=True)   # dọn venv hỏng dở của lần trước
    assert run([sys.executable, "-m", "uv", "venv", "--seed", "--clear", VENV]) == 0
print("venv:", PY)


def pipi(*args):
    assert run([sys.executable, "-m", "uv", "pip", "install", "-q",
                "--python", PY, *args]) == 0, f"cài thất bại: {args[0]}"


plan = json.loads(subprocess.run(
    [sys.executable, "tools/pick_flash_attn.py", "--plan"],
    capture_output=True, text=True, check=True).stdout)
print(json.dumps(plan, indent=2))

pipi(f"torch=={plan['torch']}", "torchvision", "--index-url", plan["index_url"])
pipi(plan["wheel"])
# Bản prerelease mà Tencent phát triển trên đó (4.57.1.dev0). Ghim để giữ được
# packing, xdrope và bố cục module gốc — xem CUSTOM_SFT.md mục "Đường ghim".
# Cài transformers TRƯỚC: 4.57 chốt tokenizers<0.23, để resolver tự chọn.
pipi("git+https://github.com/huggingface/transformers"
     "@82a06db03535c49aa987719ed0746a76093b1ec4")
pipi("accelerate", "deepspeed", "safetensors", "datasets",
     "sentencepiece", "pillow", "opencv-python-headless", "numpy",
     "einops", "tensorboard", "tqdm", "binpacking", "lmdb", "huggingface_hub>=0.34")
```

Kiểm ngay sau khi cài — ba thứ này quyết định `PACKING=1` chạy được hay không:

```python
!$PY -c "
import dataclasses, transformers as T
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import apply_rotary_pos_emb_xdrope
from transformers import TrainingArguments as A
f = {x.name for x in dataclasses.fields(A)}
print('transformers', T.__version__, '| xdrope OK')
print('warmup_ratio', 'warmup_ratio' in f, '| logging_dir', 'logging_dir' in f)"
```

Phải ra `transformers 4.57.1.dev0 | xdrope OK` và cả hai flag `True`.
`ImportError` ở dòng import nghĩa là pin không ăn — kiểm lại xem venv có sẵn bản
transformers khác không.

**Tại sao `uv` chứ không phải `python -m venv`.** Debian/Ubuntu tách `ensurepip`
ra gói `python3-venv` riêng, ảnh Colab không cài, nên stdlib venv thoát mã 1 —
và tệ hơn, nó **để lại một thư mục venv hỏng dở**, làm lần tạo kế tiếp báo
"A virtual environment already exists". uv tự mang pip nên không đụng tới
`ensurepip`, và `uv pip` kéo ~6 GB torch nhanh hơn `pip` đáng kể.

Cell chạy lại được nhiều lần: đã có `$VENV/bin/python` thì bỏ qua bước tạo,
không cài lại từ đầu; chưa có thì xoá sạch thư mục cũ rồi tạo với `--clear`.

Muốn giữ stdlib venv thì cài gói còn thiếu trước:

```bash
!apt-get install -qq -y python3.12-venv
```

Mọi lệnh cài **in stdout/stderr khi lỗi** thay vì chỉ ném `CalledProcessError`
không kèm thông tin gì.

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

**Từ đây mọi lệnh gọi script đều truyền thẳng interpreter của venv** —
`PYTHON=$VENV/bin/python`, và thêm `TORCHRUN=$VENV/bin/torchrun` cho bước train.
Trước đây chỗ này dùng `PATH=$VENV/bin:$PATH`; cách đó im lặng rơi về Python hệ
thống nếu prefix bị rớt, và lỗi chỉ lộ ra dưới dạng
`ModuleNotFoundError: No module named 'binpacking'` từ giữa file packer.

`pack_data.sh` và `sft_base_1gpu.sh` giờ kiểm tra interpreter trước khi chạy:

```
[ERROR] python (/usr/bin/python3) thiếu binpacking / transformers / Pillow.
        Dùng venv thì truyền interpreter của nó vào:
            PYTHON=/content/venv/bin/python bash scripts/pack_data.sh
```

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

Prompt: dùng `configs/ocr_prompt.md` **trong repo này**, không trích từ README
dataset. Prompt là một phần cấu hình của lần train — nó phải được version cùng
code, và phải khớp từng chữ với prompt lúc suy luận (model chỉ thấy đúng một
chuỗi prompt suốt quá trình train).

```bash
!head -3 /content/hyocr/configs/ocr_prompt.md
```

## 6. Convert + pack

```bash
!$VENV/bin/python tools/makedata_to_hyocr.py \
    --root /content/dataset \
    --prompt-file /content/hyocr/configs/ocr_prompt.md \
    --out-dir /content/hyocr/data/raw \
    --data-list /content/hyocr/data/data_list.txt
```

Ra 1.013 dòng train / 114 dòng validation, `thiếu ảnh` phải là 0.

```bash
!PYTHON=$VENV/bin/python \
 MODEL_PATH=/content/HunyuanOCR \
 INPUT_LIST=/content/hyocr/data/data_list.txt \
 PACK_OUTPUT=/content/hyocr/data/packed/train_$PACK_LEN.jsonl \
 PACK_LEN=$PACK_LEN \
 NUM_PROCESSES=2 THREADS_PER_PROCESS=4 \
 FOREGROUND=1 \
     bash scripts/pack_data.sh
```

`NUM_PROCESSES=2` chứ không phải 32: Colab chỉ có 12 vCPU và mỗi process load
một bản processor riêng. Bước này mất 10–20 phút vì phải mở từng ảnh để đếm
vision token.

> Thấy `[ERROR] /bin/python` nghĩa là `$VENV` rỗng — biến Python đó không còn
> trong namespace của kernel (thường vì kernel đã restart, hoặc bạn chưa chạy
> cell **Cấu hình**). IPython không thay thế được thì shell nhận biến rỗng và
> ghép ra `/bin/python`. Chạy lại cell Cấu hình, hoặc gõ thẳng
> `/content/venv/bin/python`.
>
> Viết `$PACK_LEN` **không có ngoặc nhọn**. Trong dòng `!` của IPython, `{...}`
> là cú pháp nội suy biểu thức Python, nên `${PACK_LEN}` thành `$` + `16384`;
> shell đọc `$1` là tham số vị trí (rỗng) và bạn nhận được
> `train_6384.jsonl`.

Kiểm tra:

```python
import json
n = t = 0
for line in open(f"/content/hyocr/data/packed/train_{PACK_LEN}.jsonl"):
    p = json.loads(line); n += 1; t += sum(x["num_tokens"] for x in p)
print(f"{n} pack, trung bình {t/n:.0f} token/pack")
```

## 7. Train

Trên transformers ghim `@82a06db` thì **bật packing** — đó là lý do ghim:

```bash
!PYTHON=/content/venv/bin/python \
 TORCHRUN=/content/venv/bin/torchrun \
 MODEL_PATH=/content/HunyuanOCR \
 TRAIN_DATA=/content/hyocr/data/packed/train_$PACK_LEN.jsonl \
 PACKING=1 \
 GRAD_ACCUM=4 \
 PACK_LEN=$PACK_LEN \
 RUN_NAME=colab_run \
 SAVE_STEPS=25 \
     bash scripts/sft_base_1gpu.sh
```

`GRAD_ACCUM=4` chứ không phải 16: một pack 16384 token chứa cỡ 4 trang, nên
4 accum × ~4 trang ≈ batch hiệu dụng 16 trang, xấp xỉ cấu hình không pack.

Số step phụ thuộc số pack — đọc từ bước kiểm tra ở mục 6:
`ceil(n_pack / 4) × EPOCHS`. Với ~250 pack và 5 epoch là khoảng 315 step.

Bản không pack (transformers phát hành ≥ 5.13) vẫn chạy được, chỉ chậm hơn:

```bash
 TRAIN_DATA=/content/hyocr/data/flat/train.jsonl  PACKING=0  GRAD_ACCUM=16
```

Chạy đồng bộ Drive ở nền trước đó:

```python
import subprocess
subprocess.Popen(
    "while true; do rsync -a --delete /content/hyocr/output/ "
    "/content/drive/MyDrive/hyocr_sft/; sleep 600; done",
    shell=True)
```

`TRAIN_DATA` là JSONL `--flat`, **không** phải file packed — xem
[`CUSTOM_SFT.md`](./CUSTOM_SFT.md) mục "Packing không dùng được trên
transformers phát hành".

Nhìn vài dòng log đầu để biết dữ liệu có vào đúng không:

| Dấu hiệu | Nghĩa |
|---|---|
| `loss` khoảng 0,2–3,0 | bình thường. HunyuanOCR vốn đã là model OCR nên loss khởi đầu thấp là đúng |
| `loss` đúng `0.0` hoặc `nan` | `labels` bị che hết — chỗ cắt prompt trong `_process_single_item` sai |
| `Skipping item` | processor trả thiếu key nào đó, mẫu bị bỏ âm thầm |

Loss khởi đầu thấp cũng có nghĩa **biên cải thiện hẹp**: phần lớn thứ học được là
quy ước Markdown và bố cục biểu mẫu VN, không phải khả năng đọc chữ. Với 1.007
mẫu, 5 epoch dễ overfit — nếu loss train tụt nhanh, cân nhắc `EPOCHS=2` hoặc 3.

OOM thì hạ `PACK_LEN` (ở chế độ không pack nó là độ dài tối đa của **một trang**,
không phải một pack) xuống 8192, rồi mới tới `TUNE_VISION=False`.

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
| `ModuleNotFoundError` bất kỳ gói nào | đang chạy bằng Python hệ thống chứ không phải venv — thiếu `PYTHON=$VENV/bin/python` (và `TORCHRUN=...` khi train) |
| `pick_flash_attn.py` báo không có wheel nào | nền tảng lạ (aarch64, cp313 mới) — build from source là đường duy nhất |
| `RuntimeError: Default process group has not been initialized` | chạy `python train/train_hunyuan.py` trực tiếp. Phải qua `torchrun` (script đã lo), vì `train_hunyuan.py:182` gọi `torch.distributed.get_rank()` |
| Pack ra 0 dòng | đường dẫn ảnh trong `data/raw/*.jsonl` là tuyệt đối của máy cũ — phải chạy lại converter trên Colab |
| Eval vỡ ngay batch đầu | đừng bật `--eval_strategy steps`, xem `CUSTOM_SFT.md` mục 4 |
| Training đứng hình ở bước lưu | đang ghi checkpoint thẳng vào Drive — output phải nằm ở `/content` |
