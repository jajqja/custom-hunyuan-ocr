# Deploy HunyuanOCR đã fine-tune bằng vLLM

Runbook cho bản đã chạy được: model `newai-vn/newai-ocr-v2` (export từ
`checkpoint-300`), serve bằng vLLM qua endpoint OpenAI-compatible. Cả trên Google
Colab và trên máy riêng — cùng một công thức, chỉ khác đường dẫn.

Model **không serve được nếu chỉ `vllm serve` thẳng**. Phải qua hai bước riêng:
`export_checkpoint.py` để dựng thư mục model đủ file, và `patch_vllm_xdrope.py`
để vLLM đi đúng nhánh rotary embedding. Mục [Vì sao phải vá](#vì-sao-phải-vá)
giải thích, đọc trước khi nâng cấp vLLM.

---

## Cấu hình đã chạy được

| | |
|---|---|
| vLLM | `0.25.1` |
| transformers | `5.12.1` |
| model dir | export từ `checkpoint-*` bằng `tools/export_checkpoint.py` |
| bản vá | `tools/patch_vllm_xdrope.py` — 3 file trong vLLM |
| `--max-model-len` | 16384 |
| `--gpu-memory-utilization` | 0.85 |
| GPU | A100-80GB (model ~1B, L4-24GB cũng đủ) |

**Venv riêng, không dùng chung với venv train.** Venv train ghim torch theo wheel
flash-attn; vLLM kéo torch riêng của nó. Cài chung là hỏng cả hai.

Cả hai gói phải ghim cứng. vLLM đổi cấu trúc `rotary_embedding` và
`gpu_model_runner` khá thường xuyên, mà ba bản vá bám vào đúng những chỗ đó.

---

## Trên Google Colab

### 1. Tải model

```bash
!pip install -q -U "huggingface_hub[cli]" uv
```

```python
from huggingface_hub import login
login()          # token quyền read trên org newai-vn; repo private
```

```bash
!hf download newai-vn/newai-ocr-v2 --local-dir /content/newai-ocr-v2
!du -sh /content/newai-ocr-v2
```

Đã có `checkpoint-*` tại chỗ thì export ra trước, đừng serve thẳng checkpoint:

```bash
!python /content/hyocr/tools/export_checkpoint.py \
    --checkpoint /content/hyocr/output/colab_run/checkpoint-300 \
    --base /content/HunyuanOCR \
    --out /content/newai-ocr-v2
```

Một `checkpoint-*` của Trainer **không load lại được một mình**: nó có trọng số +
config + tokenizer nhưng thiếu image processor (`train_hunyuan.py:232` gọi
`processor.save_pretrained` đúng một lần, vào thư mục gốc, sau khi train xong).
Nó cũng kèm `optimizer.pt` / `scheduler.pt` / `rng_state.pth` cỡ vài GB chỉ để
resume. `export_checkpoint.py` lọc phần đó, lấy processor từ model gốc, và **ghi
khoá `xdrope_section` ở cấp cao nhất của config** — khoá đó là thứ làm
`uses_xdrope_dim()` trả 4 thay vì 0, xem mục dưới.

### 2. Dựng venv serve

```bash
!pkill -f "vllm serve"; rm -rf /content/vllmenv
!python -m uv venv --seed /content/vllmenv
!python -m uv pip install -q --python /content/vllmenv/bin/python \
    "vllm==0.25.1" "transformers==5.12.1"
```

`uv` chứ không phải `python -m venv`: ảnh Colab thiếu `ensurepip` nên stdlib venv
thoát mã 1 và để lại một thư mục hỏng dở, làm lần tạo kế tiếp báo "A virtual
environment already exists". `--seed` là thứ cấp pip vào venv mới.

### 3. Vá

```bash
!python /content/hyocr/tools/patch_vllm_xdrope.py --python /content/vllmenv/bin/python
```

In ra ba dòng `đã vá` và một dòng `import OK`. Có sao lưu `*.hyocr.bak` cạnh mỗi
file, `--revert` để lùi. Chạy lại nhiều lần vô hại — nó nhận dấu `# >>>
hyocr-xdrope-patch` và bỏ qua file đã vá.

### 4. Probe — 30 giây, trước khi tốn 3 phút khởi động server

```bash
!/content/vllmenv/bin/python -c "
from vllm.transformers_utils.config import get_config, uses_xdrope_dim, uses_mrope
c = get_config('/content/newai-ocr-v2', trust_remote_code=True)
print('config class   :', type(c).__name__)
print('xdrope_dim =', uses_xdrope_dim(c), '| uses_mrope =', uses_mrope(c))" 2>/dev/null
```

Cần đúng:

```
config class   : HunYuanVLConfig
xdrope_dim = 4 | uses_mrope = False
```

Đi qua `vllm.transformers_utils.config.get_config`, **không** phải
`transformers.AutoConfig`: vLLM đăng ký class config của riêng nó vào registry.
Gọi `AutoConfig` trực tiếp trên transformers 5.12 sẽ ném
`KeyError: 'hunyuan_vl'` — đó là đúng, không phải lỗi.

`uses_mrope = True` nghĩa là bản vá 3 chưa ăn. Dừng lại, đừng serve.

### 5. Serve

```bash
%%bash
cd /content/hyocr
export PATH=/content/vllmenv/bin:$PATH
MODEL_PATH=/content/newai-ocr-v2 \
SERVED_NAME=newai-ocr-v2 \
MAX_MODEL_LEN=16384 \
GPU_MEM_UTIL=0.85 \
LOG=/content/vllm.log \
bash inference/vLLM/serve.sh
```

Phải `%%bash` chứ không phải `!`: dòng `!` của IPython chỉ chạy một lệnh, không
giữ được `cd` và vòng lặp.

```bash
!for i in $(seq 1 90); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null \
     && { echo "=== READY ==="; break; }; sleep 10; done; tail -25 /content/vllm.log
```

Lần đầu 2–4 phút vì torch.compile. Chết trong lúc compile thì thêm
`EXTRA_ARGS="--enforce-eager"` — chậm hơn nhưng khởi động ngay.

---

## Trên máy riêng

Cùng công thức, không có phần Colab. Giả sử repo ở `~/hyocr`.

```bash
cd ~/hyocr

# venv serve, TÁCH khỏi venv train
python3.12 -m venv .venv-serve
.venv-serve/bin/pip install -U pip
.venv-serve/bin/pip install "vllm==0.25.1" "transformers==5.12.1"

# vá
python tools/patch_vllm_xdrope.py --python .venv-serve/bin/python

# probe
.venv-serve/bin/python -c "
from vllm.transformers_utils.config import get_config, uses_xdrope_dim, uses_mrope
c = get_config('./export/newai-ocr-v2', trust_remote_code=True)
print('xdrope_dim =', uses_xdrope_dim(c), '| uses_mrope =', uses_mrope(c))"
```

Máy riêng có `ensurepip` nên stdlib `venv` chạy được, không cần `uv`.

```bash
PATH=$PWD/.venv-serve/bin:$PATH \
MODEL_PATH=$PWD/export/newai-ocr-v2 \
SERVED_NAME=newai-ocr-v2 \
MAX_MODEL_LEN=16384 \
GPU_MEM_UTIL=0.85 \
GPU=0 PORT=8000 \
LOG=/var/log/hyocr-vllm.log \
    bash inference/vLLM/serve.sh
```

`serve.sh` chạy nền bằng `nohup` và in pid. Biến nhận được: `MODEL_PATH` (bắt
buộc), `GPU`, `PORT`, `GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `SERVED_NAME`, `LOG`,
`EXTRA_ARGS`.

### systemd

```ini
# /etc/systemd/system/hyocr-vllm.service
[Unit]
Description=HunyuanOCR vLLM (newai-ocr-v2)
After=network-online.target

[Service]
Type=exec
User=hyocr
WorkingDirectory=/opt/hyocr
Environment=PATH=/opt/hyocr/.venv-serve/bin:/usr/local/bin:/usr/bin:/bin
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/opt/hyocr/.venv-serve/bin/vllm serve /opt/hyocr/export/newai-ocr-v2 \
    --served-model-name newai-ocr-v2 \
    -tp 1 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --trust-remote-code \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Gọi thẳng `vllm serve` chứ không qua `serve.sh`: script kia `nohup ... &` rồi
thoát, systemd sẽ tưởng service chết. `Type=exec` cần tiến trình chạy tiền cảnh.

### Docker

```dockerfile
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv curl && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Ghim cứng cả hai. Nâng bản nào cũng phải chạy lại probe ở mục 4.
RUN pip install --no-cache-dir "vllm==0.25.1" "transformers==5.12.1"

COPY tools/patch_vllm_xdrope.py /opt/patch_vllm_xdrope.py
RUN python /opt/patch_vllm_xdrope.py --python /opt/venv/bin/python

COPY export/newai-ocr-v2 /model

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
ENTRYPOINT ["vllm", "serve", "/model", \
    "--served-model-name", "newai-ocr-v2", \
    "-tp", "1", \
    "--limit-mm-per-prompt", "{\"image\":4,\"video\":0}", \
    "--trust-remote-code", \
    "--port", "8000", \
    "--gpu-memory-utilization", "0.85", \
    "--max-model-len", "16384", \
    "--max-num-batched-tokens", "16384"]
```

Vá **lúc build**, không lúc chạy: image bất biến thì bản vá cũng nằm trong layer,
và `patch_vllm_xdrope.py` gọi `ast.parse` trước khi ghi nên build sẽ đỏ ngay nếu
vLLM đã đổi cấu trúc — tốt hơn là phát hiện lúc container khởi động.

```bash
docker build -t hyocr:v2 .
docker run --gpus '"device=0"' -p 8000:8000 hyocr:v2
```

---

## Gọi model

```python
import base64, json, urllib.request

PROMPT = open("configs/ocr_prompt.md", encoding="utf-8").read().strip()

def ocr(path, model="newai-ocr-v2", url="http://127.0.0.1:8000/v1/chat/completions"):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT}]}],
            "temperature": 0.0,
            "max_tokens": 4096,
            "repetition_penalty": 1.08,
        }).encode())
    return json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]

print(ocr("test.jpg"))
```

Ba chi tiết không được đổi:

- **Prompt phải khớp từng chữ với `configs/ocr_prompt.md` lúc train.** Model chỉ
  thấy đúng một chuỗi prompt trong suốt quá trình train, nên đổi câu chữ là lệch
  phân phối chứ không phải "prompt engineering".
- `temperature: 0.0` + `repetition_penalty: 1.08` — đúng công thức của
  `inference/transformers/infer_hf_8gpu.py`, để số đo so được với baseline.
- `repetition_penalty` để ở **cấp cao nhất** của JSON. `extra_body` là khái niệm
  của client OpenAI; POST JSON thô mà bọc vào đó thì server bỏ qua im lặng.

### Chấm CER cả tập

```bash
python tools/eval_checkpoint.py \
    --server http://127.0.0.1:8000/v1 --served-name newai-ocr-v2 \
    --data ./data/flat/validation.jsonl \
    --concurrency 16 \
    --out ./eval_v2.jsonl
```

Qua server nhanh hơn load tại chỗ nhiều lần nhờ continuous batching — 112 mẫu
với 16 request song song mất 3–5 phút. Muốn tập con thì `--sample 40`, **không
phải `--limit`**: JSONL ghi lần lượt theo từng config nên 40 dòng đầu toàn GCN.

Đổ ra cây .md để soi mắt:

```bash
python tools/eval_to_md.py --eval ./eval_v2.jsonl --out ./eval_v2_md \
    --image-root /content/hf_upload=./hf_upload --label newai-ocr-v2
```

Dừng server: `pkill -f "vllm serve"`

---

## Vì sao phải vá

Bốn mắt xích, đứt cả bốn vì cùng một nguyên nhân.

Config gốc của HunyuanOCR khai báo:

```json
"rope_scaling": {"type": "xdrope", "xdrope_section": [16,16,16,16], "alpha": 1000.0}
```

`xdrope` là rotary embedding **4 trục** (`xdrope_section` có 4 phần tử). Nhưng
transformers — **cả 5.12 lẫn 5.13+**, không bản nào thoát — chuẩn hoá khối đó
**lúc đọc**:

```
[transformers] Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='xdrope'
[transformers] Unrecognized keys in `rope_parameters` for 'rope_type'='dynamic': {'mrope_section', 'alpha', ...}
```

Nó gặp `xdrope`, không có validator cho tên đó, nên đổi thành `dynamic` và đổi
tên `xdrope_section` → `mrope_section`:

```python
{'rope_type': 'dynamic', 'mrope_section': [16,16,16,16], 'alpha': 1000.0, 'type': 'dynamic'}
```

Từ đó bốn chỗ trong vLLM trượt:

| # | Chỗ | Hỏng thế nào | Sửa bằng |
|---|---|---|---|
| 1 | `uses_xdrope_dim()` | tìm khoá `xdrope_section`, không còn → trả 0 | `export_checkpoint.py` ghi `xdrope_section` **cấp cao nhất** của config |
| 2 | `get_rope()` | phân nhánh theo `rope_type`; `dynamic`+`alpha` → `DynamicNTKAlphaRotaryEmbedding`, tức RoPE **1 chiều** cho model cần 4 trục | vá 1 |
| 3 | `hunyuan_vision.get_xdrope_input_positions()` | đọc `hf_config.rope_scaling["xdrope_section"]`, `rope_scaling` giờ là `None` → `TypeError` | vá 2 |
| 4 | `gpu_model_runner.py:1034/1040/2229` | kiểm `uses_mrope` **trước** `uses_xdrope_dim`; `mrope_section` tồn tại nên mrope thắng → cấp buffer position **3 dòng** cho model cần 4 | vá 3 |

Triệu chứng nếu bỏ qua:

```
# torch.compile
view(..., 3*s18, -1, 128)          shape error ở profile_run

# --enforce-eager
rotary_embedding: query, key and positions must have the same batch_size and seq_len
```

`--enforce-eager` **không sửa được gì** — nó chỉ đổi chỗ báo lỗi từ torch.compile
sang CUDA kernel. Hạ vLLM cũng không: normalization nằm ở transformers, không ở
vLLM.

### Ba bản vá làm gì

| File trong vLLM | Sửa |
|---|---|
| `model_executor/layers/rotary_embedding/__init__.py` | sau `scaling_type = rope_parameters.get("rope_type","default")`, ánh xạ `dynamic`+`alpha`+`mrope_section` → `xdrope` |
| `model_executor/models/hunyuan_vision.py` | đọc section mềm hơn: `rope_scaling["xdrope_section"]` → `xdrope_section` cấp cao nhất → `rope_parameters["mrope_section"]` |
| `transformers_utils/config.py` | `uses_mrope()` trả `False` khi `uses_xdrope_dim(config) > 0` |

Nhánh `xdrope` **vẫn còn nguyên** trong vLLM — có `XDRotaryEmbedding`, có
`get_xdrope_input_positions()`, có buffer `xdrope_positions` riêng. Chỉ là không
còn config nào đi vào được. Bản vá khôi phục đường đi đó, **không đụng tới toán**.

`patch_vllm_xdrope.py` chạy `ast.parse` trên kết quả và **từ chối ghi** nếu không
compile. Thụt lề sai làm file vẫn ghi được nhưng vỡ lúc import — đã bị một lần vì
dùng điểm neo thụt 4 space cho dòng thật thụt 8 space, khiến `str.replace` khớp
giữa dòng.

### Hai điều dễ nhầm

**Dùng thẳng config của model gốc không giúp gì.** Đã kiểm: config gốc và config
export load ra `rope_parameters` **giống hệt nhau**. Normalization xảy ra **lúc
đọc**, không phải lúc ghi. Vì vậy phần khôi phục `rope_scaling` trong
`export_checkpoint.py` là vô tác dụng — thứ có tác dụng là khoá
`xdrope_section` cấp cao nhất mà nó thêm vào.

**Hạ transformers xuống dưới 5.13 không tránh được normalization.** `hunyuan_vl`
chỉ vào transformers từ 5.13.0 (5.11/5.12 trả 404), nên ở 5.12 thì vLLM dùng class
config **của chính nó** (`vllm/transformers_utils/configs/hunyuan_vl.py`) — probe
xác nhận `config class: HunYuanVLConfig`. Nhưng class đó vẫn kế thừa
`PretrainedConfig`, và `RotaryEmbeddingConfigMixin` của 5.12 đã chuẩn hoá rope
rồi. Cái mà 5.12 tránh được là một lỗi khác: vLLM 0.25.1 gọi
`AutoImageProcessor.register("<chuỗi>", cls)`, còn transformers 5.15 đòi class →
`AttributeError: 'str' object has no attribute '__module__'`. Đó là lý do ghim
5.12.1, không phải vì rope.

---

## Sự cố đã gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `KeyError: 'hunyuan_vl'` từ `AutoConfig` | probe gọi `transformers.AutoConfig` thay vì `vllm...get_config` | đúng như vậy, không phải lỗi — dùng `get_config` |
| `AttributeError: 'str' object has no attribute '__module__'` | vLLM 0.25.1 + transformers ≥ 5.13 | ghim `transformers==5.12.1` |
| `xdrope_dim = 0` | model dir chưa qua `export_checkpoint.py` | export lại |
| `uses_mrope = True` | vá 3 chưa ăn | chạy `patch_vllm_xdrope.py`, xem có dòng `đã vá` |
| `rotary_embedding: query, key and positions must have the same batch_size` | vá 1 chưa ăn | như trên |
| `view(..., 3*s18, -1, 128)` ở profile_run | vá 1 hoặc 3 chưa ăn | như trên; `--enforce-eager` **không** phải cách sửa |
| `không tìm thấy điểm neo trong ...` | vLLM đã đổi cấu trúc | vá tay, cập nhật điểm neo trong `patch_vllm_xdrope.py` |
| flashinfer version mismatch | cài 0.25.1 đè lên bản mới hơn trong cùng venv | venv sạch; nếu vẫn còn thì `FLASHINFER_DISABLE_VERSION_CHECK=1` — vô hại vì backend là FLASH_ATTN và sinh greedy |
| server im, không có ảnh nào ra | thiếu image processor trong model dir | `export_checkpoint.py` lấy từ `--base` |
| `ModuleNotFoundError` lúc gọi script | dùng `python` hệ thống thay vì interpreter của venv | truyền `--python /path/venv/bin/python` |

## Khi nâng cấp vLLM

Ba bản vá bám vào cấu trúc file cụ thể. Thứ tự làm:

1. `patch_vllm_xdrope.py --python ... --revert` trên venv cũ, hoặc dựng venv mới.
2. Cài bản vLLM mới, chạy `patch_vllm_xdrope.py`. Báo `không tìm thấy điểm neo`
   thì đọc file thật, cập nhật `ANCHOR` / `V_OLD` / `C_OLD`.
3. Probe. `xdrope_dim = 4 | uses_mrope = False` mới đi tiếp.
4. Serve, gọi thử một ảnh, rồi chấm lại CER trên đúng tập validation cũ.

Bước 4 không bỏ được: sai nhánh rotary vẫn có thể **chạy mà không crash** và chỉ
lộ ra ở chất lượng đầu ra. Mốc để so: **CER 0.0286** trên val 112 mẫu.

Đúng ra thì đây là bug của vLLM ở chỗ `uses_mrope` được kiểm trước
`uses_xdrope_dim` — đáng mở issue upstream kèm ba bản vá này.
