# Chạy SFT trên Google Colab

Bản Colab của [`CUSTOM_SFT.md`](./CUSTOM_SFT.md). Notebook tương ứng:
[`notebooks/hyocr_sft_colab.ipynb`](../notebooks/hyocr_sft_colab.ipynb) — tải lên
Colab rồi chạy từ trên xuống.

Doc này chỉ tới hết bước train. Serve model đã train bằng vLLM — trên Colab hay
trên máy riêng — ở [`DEPLOY_VLLM.md`](./DEPLOY_VLLM.md).

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

Ở chế độ không pack (`PACKING=0`, xem mục 7), `PACK_LEN` là **độ dài tối đa của
một trang** chứ không phải của một pack — nó đi vào `--packed_max_length`, và
`VLDataCollator` dùng nó làm mức cắt. Một trang A4 200 DPI tốn phần lớn ngân sách
token vào vision token, nên 16384 là dư cho một trang ở 80GB.

40GB thì 8192; ép lên 16384 sẽ OOM giữa chừng chứ không phải ngay step đầu.

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
# `hunyuan_vl` chỉ có từ transformers 5.13.0 (5.11/5.12 chưa có), nên phải là
# bản phát hành mới. Đừng ghim @82a06db: bản đó thiếu HunYuanVLVisionTransformer
# nên train/trainer.py không import được — xem CUSTOM_SFT.md.
pipi("transformers>=5.13", "accelerate", "deepspeed", "safetensors", "datasets",
     "sentencepiece", "tokenizers", "pillow", "opencv-python-headless", "numpy",
     "einops", "tensorboard", "tqdm", "binpacking", "lmdb", "huggingface_hub>=0.34")
```

Kiểm ngay sau khi cài:

```python
!$PY -c "
import transformers as T
from transformers import HunYuanVLForConditionalGeneration
from train.trainer import replace_hunyuanocr_attention_class
print('transformers', T.__version__, '| import OK')"
```

`ModuleNotFoundError: transformers.models.hunyuan_vl` nghĩa là bản quá cũ.

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

# dataset (~600 MB), private — token phải có quyền read trên org newai-vn
!$VENV/bin/hf download newai-vn/vietnamese-doc-ocr --repo-type dataset \
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

## 6. Convert sang JSONL phẳng

```bash
!$VENV/bin/python tools/makedata_to_hyocr.py \
    --root /content/dataset \
    --prompt-file /content/hyocr/configs/ocr_prompt.md \
    --out-dir /content/hyocr/data/flat --flat
```

Ra 1.015 dòng train / 112 dòng validation, `thiếu ảnh` phải là 0.

`--flat` xuất `{"image","question","answer"}` — schema mà `VLDataset` đọc trực
tiếp khi `is_packed=False`. Không cần bước pack: packing chỉ chạy được trên bản
transformers prerelease đã ngừng dùng (xem
[`CUSTOM_SFT.md`](./CUSTOM_SFT.md) mục "Packing không dùng được trên transformers
phát hành"), và với 1.015 mẫu thì nó không đáng.

```python
import json
for sp in ("train", "validation"):
    rows = [json.loads(l) for l in open(f"/content/hyocr/data/flat/{sp}.jsonl")]
    print(f"{sp:11} {len(rows):5} mẫu | nhãn rỗng {sum(not r['answer'].strip() for r in rows)}")

!mkdir -p /content/drive/MyDrive/hyocr_sft/flat \
    && cp /content/hyocr/data/flat/*.jsonl /content/drive/MyDrive/hyocr_sft/flat/
```

Sao lưu sang Drive luôn — chỉ vài MB, và mất session thì không phải convert lại.

> Thấy `[ERROR] /bin/python` nghĩa là `$VENV` rỗng — biến Python đó không còn
> trong namespace của kernel (thường vì kernel đã restart, hoặc bạn chưa chạy
> cell **Cấu hình**). IPython không thay thế được thì shell nhận biến rỗng và
> ghép ra `/bin/python`.
>
> Viết `$PACK_LEN` **không có ngoặc nhọn**. Trong dòng `!` của IPython, `{...}`
> là cú pháp nội suy biểu thức Python, nên `${PACK_LEN}` thành `$` + `16384`.

## 7. Train

Cấu hình đã chạy thật trên A100-80GB (transformers 5.15, torch 2.10):

```bash
!PYTHON=/content/venv/bin/python \
 TORCHRUN=/content/venv/bin/torchrun \
 MODEL_PATH=/content/HunyuanOCR \
 TRAIN_DATA=/content/hyocr/data/flat/train.jsonl \
 PACKING=0 \
 GRAD_ACCUM=16 \
 PACK_LEN=16384 \
 RUN_NAME=colab_run \
 SAVE_STEPS=25 \
     bash scripts/sft_base_1gpu.sh
```

320 step (1.015 mẫu / 16 accum × 5 epoch).

Không có validation trong lúc train: `Trainer` chỉ nhận **một** collator dùng cho
cả train và eval, nên muốn eval loss phải thêm
`EVAL_DATA=/content/hyocr/data/flat/validation.jsonl EVAL_STEPS=32`. Nhưng thứ
quyết định chọn checkpoint nào là CER sinh thật, đo sau bằng `eval_checkpoint.py`
— eval loss mỗi 32 step chỉ tốn thời gian.

Chạy đồng bộ Drive ở nền trước đó:

```python
import subprocess
subprocess.Popen(
    "mkdir -p /content/drive/MyDrive/hyocr_sft/output; "
    "while true; do rsync -a --delete /content/hyocr/output/ "
    "/content/drive/MyDrive/hyocr_sft/output/; sleep 600; done",
    shell=True)
```

> Đích phải là `hyocr_sft/output/`, **không** phải `hyocr_sft/`. `--delete` xoá
> mọi thứ ở đích mà nguồn không có, nên trỏ vào thư mục gốc sẽ xoá luôn
> `hyocr_sft/flat/` — bản sao lưu JSONL của bước 6.

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
quy ước Markdown và bố cục biểu mẫu VN, không phải khả năng đọc chữ. Với 1.015
mẫu, 5 epoch dễ overfit — nếu loss train tụt nhanh, cân nhắc `EPOCHS=2` hoặc 3.

OOM thì hạ `PACK_LEN` (ở chế độ không pack nó là độ dài tối đa của **một trang**,
không phải một pack) xuống 8192, rồi mới tới `TUNE_VISION=False`.

## 8. Chạy lại sau khi mất session

`train_hunyuan.py:221` tự resume nếu `output_dir` có `checkpoint-*`. Nên sau khi
làm lại bước 3–6, copy checkpoint từ Drive về rồi chạy lại đúng lệnh bước 7 với
**cùng `RUN_NAME`**:

```bash
!mkdir -p /content/hyocr/output /content/hyocr/data/flat
!rsync -a /content/drive/MyDrive/hyocr_sft/output/ /content/hyocr/output/
!rsync -a /content/drive/MyDrive/hyocr_sft/flat/   /content/hyocr/data/flat/
```

Bước 6 không cần chạy lại: JSONL phẳng chỉ vài MB nên đã được sao lưu sang
`hyocr_sft/flat/` ngay sau khi convert. Nhưng **tải lại dataset thì vẫn phải** —
đường dẫn ảnh trong JSONL là tuyệt đối (`/content/dataset/...`).

## 9. Lấy model ra

```bash
!$VENV/bin/hf upload newai-vn/newai-ocr-v2 /content/hyocr/output/colab_run . \
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
| `FileNotFoundError` một đường dẫn ảnh lúc train | `data/flat/*.jsonl` giữ đường dẫn **tuyệt đối**. Khôi phục JSONL từ Drive mà chưa tải lại dataset về `/content/dataset` thì ảnh không tồn tại — tải dataset trước |
| `hyocr_sft/flat/` biến mất trên Drive | vòng `rsync --delete` trỏ vào `hyocr_sft/` thay vì `hyocr_sft/output/` — xem cảnh báo ở bước 7 |
| Eval vỡ ngay batch đầu | đừng bật `--eval_strategy steps`, xem `CUSTOM_SFT.md` mục 4 |
| Training đứng hình ở bước lưu | đang ghi checkpoint thẳng vào Drive — output phải nằm ở `/content` |
