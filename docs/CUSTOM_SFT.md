# SFT HunyuanOCR-1.5 base trên dữ liệu tiếng Việt — 1 GPU 80GB

Runbook cho fork này. Nguồn dữ liệu: `makedata/hf_upload` (1.007 train / 120
validation, 4 config: GCN, Documents, Others, Handwriting).

Chạy trên Google Colab thì xem [`COLAB.md`](./COLAB.md) — cùng quy trình nhưng
đã tính tới A100-40GB, flash-attn wheel, và chuyện mất session.

Tất cả script upstream vẫn giữ nguyên hành vi cũ; phần riêng của fork là
`scripts/env_single.sh`, `scripts/sft_base_1gpu.sh`, `tools/makedata_to_hyocr.py`
và cờ `--allow-empty-answer` trong `tools/pipeline_count_and_pack.py`.

---

## 0. Môi trường

Dựng môi trường riêng, đừng cài vào Python hệ thống — bước dưới ghim một bản
torch cụ thể và sẽ kéo tụt các gói khác nếu cài chung.

```bash
python3.12 -m venv .venv           # hoặc: conda create -n hyocr python=3.12 -y
source .venv/bin/activate
pip install -U pip setuptools wheel
```

**Cài torch trước, và phải là bản có wheel flash-attn.**
`train/trainer.py:3` import `flash_attn_varlen_func` ngay đầu module, còn
`train_hunyuan.py` truyền cứng `attn_implementation="flash_attention_2"` —
không có đường lùi sang eager, thiếu là chết lúc load model chứ không phải lúc
train. Wheel flash-attn build riêng cho từng tổ hợp (CUDA major, torch minor,
cxx11 ABI, Python, arch) và torch luôn ra trước wheel vài tuần, nên **cài torch
mới nhất rồi mới tìm wheel là ngược**. Hỏi trước:

```bash
python tools/pick_flash_attn.py --plan
# {"torch": "2.10.0", "index_url": "https://download.pytorch.org/whl/cu128",
#  "wheel": "https://github.com/Dao-AILab/.../flash_attn-2.8.1+cu12torch2.10...whl"}
```

rồi cài đúng bộ đó:

```bash
pip install "torch==2.10.0" torchvision --index-url https://download.pytorch.org/whl/cu128
python tools/pick_flash_attn.py --install     # tự chọn wheel khớp torch vừa cài
```

Còn lại:

```bash
pip install -r requirements.txt
pip install binpacking lmdb
```

Đã có sẵn torch không đổi được thì `python tools/pick_flash_attn.py --install`
vẫn dùng được — không có wheel khớp thì nó in ra bản torch gần nhất có, hoặc
lùi về `pip install flash-attn --no-build-isolation` (build 40–60 phút).

Tải model gốc:

```bash
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR
export MODEL_PATH=$PWD/HunyuanOCR
```

## 1. Chuyển dữ liệu sang định dạng raw JSONL

```bash
python tools/makedata_to_hyocr.py \
    --root /home/jaqja/New_AI/makedata/hf_upload \
    --prompt-file /home/jaqja/New_AI/makedata/ocr_prompt.md \
    --out-dir ./data/raw \
    --data-list ./data/data_list.txt
```

Sinh ra `data/raw/train.jsonl` (1.007 dòng) và `data/raw/validation.jsonl`
(120 dòng), mỗi dòng:

```json
{"img_path_sh": "/abs/000.jpg", "conv": [{"question": "<prompt>", "answer": "<markdown>"}]}
```

> **`docs/data_format.md` của upstream mô tả sai.** Doc ghi schema
> `{"image_path": [...], "conversations": [{"from","value"}]}` và packed output
> `{"packed_samples", "cu_seqlens", "total_tokens"}`. Không có dòng code nào
> trong repo đọc các key đó. `tools/pipeline_count_and_pack.py:157-184` đọc
> `img_path_sh` / `img_path_cq` + `conv[0].question` / `conv[0].answer`, và
> `train/data_processor.py:153-189` đọc `item["image"]` / `["question"]` /
> `["answer"]` với mỗi dòng packed là **một JSON array**. Converter bám theo code.

Prompt nằm ở `question`, không phải `system`: hàm pack không chuyển
`--system-prompt` vào bản ghi packed, nên system prompt được **tính token lúc
pack rồi biến mất lúc train** — đặt prompt ở đó là lệch dữ liệu.

## 2. Pack

```bash
MODEL_PATH=$MODEL_PATH \
INPUT_LIST=./data/data_list.txt \
PACK_OUTPUT=./data/packed/train_16384.jsonl \
PACK_LEN=16384 \
NUM_PROCESSES=8 THREADS_PER_PROCESS=8 \
FOREGROUND=1 \
    bash scripts/pack_data.sh
```

`ALLOW_EMPTY=1` là mặc định của fork. Upstream loại thẳng mọi mẫu có `answer`
rỗng (`if not question or not answer: return None`) — với bộ này là **25 trang
trắng** cố ý gán nhãn rỗng để dạy model im lặng. Loại chúng đi thì dạy đúng điều
ngược lại. Đặt `ALLOW_EMPTY=0` nếu muốn hành vi upstream.

Kiểm tra kết quả:

```bash
python -c "
import json
n=t=0
for line in open('./data/packed/train_16384.jsonl'):
    p=json.loads(line); n+=1; t+=sum(x['num_tokens'] for x in p)
print(f'{n} pack, trung bình {t/n:.0f} token/pack')"
```

Mỗi trang A4 200 DPI (1656×2339 ≈ 3,87 Mpx, dưới ngưỡng `max_pixels`
= 2048² nên **không bị resize**) tốn phần lớn ngân sách token là vision token.
Nếu số pack ra quá ít (mỗi pack chỉ 2–3 trang) thì cân nhắc `--max-pixels`
nhỏ hơn khi pack — đổi lại chữ nhỏ trên giấy sẽ mờ đi.

## 3. Train

```bash
MODEL_PATH=$MODEL_PATH \
TRAIN_DATA=./data/packed/train_16384.jsonl \
PACK_LEN=16384 \
    bash scripts/sft_base_1gpu.sh
```

Khác gì so với `scripts/sft_base.sh`:

| | `sft_base.sh` | `sft_base_1gpu.sh` |
|---|---|---|
| GPU | 8 | **1** |
| Env | `env_common.sh` (IB, `bond1`, 8 HCA mlx5) | **`env_single.sh`** (không IB) |
| `packed_max_length` | 20480 | **16384** |
| `GRAD_ACCUM` | 1 | **4** |
| `SAVE_STEPS` | 200 | **50** |
| `warmup_ratio` | 0.03 | 0.03 |
| DeepSpeed | không | tuỳ chọn qua `DEEPSPEED=scripts/zero2.json` |

Model chỉ ~1B tham số nên ở bf16 + `gradient_checkpointing` thì 80GB thừa sức
chạy full SFT (vision + MLP + LLM) mà **không cần ZeRO**. Biến còn lại ăn VRAM là
`PACK_LEN`; upstream ghi rằng 20480 vẫn OOM được trên 80GB nên fork lấy 16384 làm
mặc định. Nếu `nvidia-smi` cho thấy còn nhiều headroom, nâng dần:

```bash
PACK_LEN=20480 bash scripts/sft_base_1gpu.sh    # nhớ pack lại data cùng độ dài
```

`PACK_LEN` lúc train **phải bằng** `PACK_LEN` lúc pack.

Các nút khác:

```bash
TUNE_VISION=False ...     # đóng băng vision tower, nhanh hơn ~15%
EPOCHS=3 LR=1e-5 ...
DEEPSPEED=scripts/zero2.json ...   # chỉ khi OOM sau khi đã hạ PACK_LEN
```

Train tự resume: nếu `output_dir` đã có `checkpoint-*` thì
`train_hunyuan.py:221` gọi `trainer.train(resume_from_checkpoint=True)`. Muốn
chạy lại từ đầu thì đổi `RUN_NAME` hoặc xoá thư mục.

## 4. Đánh giá

**Chỉ bật được khi `PACKING=0`.** `train_hunyuan.py:198-205` luôn tạo
`eval_dataset` với `is_packed=False`; nếu packing bật thì collator là
`PackedVLDataCollator` (chờ list-of-list) và vỡ ngay batch eval đầu. Ở chế độ
không pack thì cả train lẫn eval dùng chung `VLDataCollator`, chạy được.

```bash
EVAL_DATA=./data/flat/validation.jsonl EVAL_STEPS=32 \
PACKING=0 TRAIN_DATA=./data/flat/train.jsonl GRAD_ACCUM=16 \
    bash scripts/sft_base_1gpu.sh
```

Chọn `EVAL_STEPS` theo số step một epoch, không theo con số tròn:

| `EVAL_STEPS` | Số lần eval (315 step) | Thêm bao lâu |
|---:|---:|---|
| 16 | 20 | ~20 phút |
| **32** | **10** (2 lần/epoch) | **~10 phút** |
| 63 | 5 (1 lần/epoch) | ~5 phút |

Mỗi lần eval là 120 forward pass ở batch 1, cỡ **40–60 giây**. `EVAL_STEPS=32`
cho đủ 10 điểm để nhìn ra chỗ val loss quay đầu — đó là toàn bộ lý do bật eval
với bộ 1.007 mẫu, nơi 5 epoch rất dễ overfit. Dày hơn 16 chỉ tốn thời gian.

Muốn `--load_best_model_at_end` thì `SAVE_STEPS` phải là bội của `EVAL_STEPS`
(đặt cả hai bằng 32); script không bật sẵn.

`eval_loss` chỉ là proxy — nó đo teacher-forced next-token, không đo được
model tự sinh ra Markdown đúng cấu trúc hay không.

### Chấm thật bằng generation

```bash
python tools/eval_checkpoint.py \
    --model ./output/colab_run/checkpoint-300 \
    --processor ./HunyuanOCR \
    --data ./data/flat/validation.jsonl \
    --out ./output/eval_ckpt300.jsonl
```

`--processor` **bắt buộc** khi chấm một `checkpoint-*`: Trainer chỉ lưu tokenizer
vào checkpoint, image processor chỉ được ghi vào thư mục gốc sau khi train xong
(`train_hunyuan.py` gọi `processor.save_pretrained` đúng một lần ở cuối). Thiếu
nó thì `AutoProcessor.from_pretrained` sẽ lỗi hoặc lấy nhầm cấu hình ảnh.

Sinh theo đúng công thức của `inference/transformers/infer_hf_8gpu.py` (greedy,
`repetition_penalty=1.08`) để số đo so được với baseline. In CER/WER tách theo
từng config, và **chấm riêng nhãn rỗng bằng exact-match** vì CER không định
nghĩa được khi reference rỗng.

`--limit 20` để thử nhanh trước khi chạy đủ 120 mẫu. Cài `rapidfuzz` cho phần
tính khoảng cách nhanh hơn ~100 lần; thiếu thì tự động dùng bản Python thuần.

Muốn so nhiều checkpoint thì chạy lại với `--model` khác nhau — mỗi lần cỡ
5–10 phút cho 120 mẫu.

## 5. Những chỗ đã sửa của upstream

| File | Sửa |
|---|---|
| `scripts/sft_base.sh:106`, `sft_dflash.sh:127`, `sft_dflash_finetune.sh:140` | `${entry_file} "${args}"` → bỏ nháy. Có nháy thì toàn bộ flag thành **một** phần tử `argv`, `HfArgumentParser.parse_args_into_dataclasses()` chết ngay. |
| `tools/pipeline_count_and_pack.py` | thêm `--allow-empty-answer` |
| `scripts/pack_data.sh` | thêm `ALLOW_EMPTY`, `FOREGROUND`, `PYTHON` |
| `scripts/sft_base_1gpu.sh` | thêm `PYTHON` / `TORCHRUN`, preflight, lọc flag qua `tools/filter_train_args.py` |
| `train/trainer.py` | import `apply_rotary_pos_emb_xdrope` thành tuỳ chọn; bỏ monkeypatch trên transformers ≥ 5.13 |
| `train/train_hunyuan.py` | `resolve_submodules()` — bố cục module đổi tên ở transformers 5.13 |
| `train/data_processor.py` | `position_ids` thành tuỳ chọn; viết lại `VLDataCollator` |
| `scripts/sft_base_1gpu.sh` | thêm `PACKING=0` |
| `tools/makedata_to_hyocr.py` | thêm `--flat` |

### Serve bằng vLLM: phải vá `get_rope`

Config gốc HunyuanOCR khai báo `{"type": "xdrope", "xdrope_section": [...]}`.
transformers 5.x chuẩn hoá nó khi load thành
`{"rope_type": "dynamic", "mrope_section": [...]}`, và vLLM đọc **bản đã chuẩn
hoá** đó, nên `get_rope()` rơi vào nhánh `dynamic` + `alpha` → RoPE 1 chiều cho
một model dùng position 4 trục. Engine chết ở `profile_run`:

```
view(..., 3*s18, -1, 128) is invalid for input of size 2048*s59
# hoặc với --enforce-eager:
rotary_embedding: query, key and positions must have the same batch_size and seq_len
```

Không sửa được bằng `config.json` — kiểm chứng: model gốc và bản đã SFT cho ra
`rope_parameters` **giống hệt nhau** sau khi transformers load. Nhánh `xdrope`
vẫn còn trong vLLM, chỉ là không config nào đi vào được nữa.

vLLM **có** hỗ trợ xdrope đầy đủ — `hunyuan_vision.get_xdrope_input_positions()`
dựng đủ 4 trục, và runner có buffer `xdrope_positions` riêng. Hỏng nằm ở khâu
nhận diện: `uses_xdrope_dim()` tìm khoá tên `xdrope_section`, mà transformers
5.x đổi thành `mrope_section` **và** dời `rope_scaling` sang `rope_parameters`
(`config.rope_scaling` trở thành `None`). Cả hai chỗ đều trượt → trả 0 → runner
cấp buffer 3 trục cho model cần 4 → `IndexError: index 3 is out of bounds`.

Cần cả hai mảnh:

```bash
python tools/export_checkpoint.py ...          # thêm xdrope_section cấp cao nhất
python tools/patch_vllm_xdrope.py --python /content/vllmenv/bin/python
```

`export_checkpoint.py` ghi `"xdrope_section": [16,16,16,16]` ở **cấp cao nhất**
của config — `PretrainedConfig` giữ nguyên khoá lạ thành thuộc tính, nên
`getattr(config, "xdrope_section")` bắt được và `uses_xdrope_dim` trả 4.

`patch_vllm_xdrope.py` vá hai chỗ trong vLLM:

| File | Sửa |
|---|---|
| `rotary_embedding/__init__.py` | `get_rope()`: `dynamic` + `mrope_section` + `alpha` → chuyển sang nhánh `xdrope` |
| `models/hunyuan_vision.py` | `xd_num = len(hf_config.rope_scaling["xdrope_section"])` → đọc mềm, `rope_scaling` giờ là `None` |
| `transformers_utils/config.py` | `uses_mrope()` trả `False` khi `uses_xdrope_dim > 0` |

Mảnh thứ ba là mảnh dễ sót nhất. Runner kiểm tra `uses_mrope` **trước**
`uses_xdrope_dim` ở cả ba chỗ chọn buffer position
(`gpu_model_runner.py:1034`, `:1040`, `:2229`). transformers tạo ra
`mrope_section` nên `uses_mrope` = True, mrope thắng, và model vẫn nhận buffer
3 dòng dù xdrope đã được nhận diện đúng — cùng một `IndexError`, không đổi.

Idempotent, sao lưu `.hyocr.bak`, gỡ bằng `--revert`, và **kiểm tra file sau khi
vá có parse được không trước khi ghi** — thụt lề sai vẫn ghi được nhưng chỉ vỡ
lúc import. Lúc deploy thì thêm cả hai dòng vào Dockerfile sau `pip install vllm`.

Các mảnh khác của môi trường serve:

| Thứ | Ràng buộc |
|---|---|
| vLLM | **≥ 0.26**. Bản 0.25.1 gọi `AutoImageProcessor.register("<chuỗi>", cls)`, transformers 5.15 đòi class → `AttributeError: 'str' object has no attribute '__module__'` |
| flashinfer | `flashinfer-cubin` mới nhất (0.6.13) không khớp `flashinfer-python` (0.6.16+) và không có bản khớp. Đặt `FLASHINFER_DISABLE_VERSION_CHECK=1`; vô hại vì attention backend là FLASH_ATTN và ta sinh greedy |
| model dir | phải qua `export_checkpoint.py` — `checkpoint-*` thiếu image processor |

### Packing không dùng được trên transformers phát hành

`PackedVLDataCollator` truyền **cu_seqlens qua đúng chỗ của `attention_mask`**
(`data_processor.py:360-366`) — chỉ có nghĩa với bản attention đã monkeypatch,
mà bản đó không còn áp dụng được (xem mục trên). Nó cũng dựng `position_ids`
dạng `[batch, 4, seq]` trong khi bản phát hành mong `[num_mrope_axes + 1, batch,
seq]`. Hai thứ này không vá lặt vặt được.

Thêm nữa, processor của transformers ≥ 5.13 **không trả `position_ids`** nữa
(model tự dựng bằng `get_rope_index`). Code cũ đọc `inputs["position_ids"]` nên
mọi mẫu đều ném `KeyError`, bị `__getitem__` nuốt thành
`[WARNING] Skipping item in pack` — **mọi pack đều rỗng, train trên không có gì**.

Nên trên transformers phát hành phải chạy không pack:

```bash
python tools/makedata_to_hyocr.py --root ... --prompt-file ... \
    --out-dir ./data/flat --flat

PACKING=0 TRAIN_DATA=./data/flat/train.jsonl GRAD_ACCUM=16 \
    bash scripts/sft_base_1gpu.sh
```

`--flat` xuất `{"image","question","answer"}` — schema mà `VLDataset` đọc trực
tiếp khi `is_packed=False`. `GRAD_ACCUM=16` để batch hiệu dụng xấp xỉ mức cũ
(một pack 16384 token chứa cỡ 4 trang, giờ mỗi step một trang).

Với 1.007 mẫu thì phần lãng phí do padding không đáng kể — packing là tối ưu
throughput cho vòng train cỡ triệu mẫu.

### Bố cục module đổi tên

| pre-merge (code gốc) | transformers ≥ 5.13 |
|---|---|
| `model.vit` | `model.model.vision_tower` |
| `model.vit.perceive` | `model.model.vision_tower.patch_merger` |
| `model.model` (phần ngôn ngữ) | `model.model.language_model` |
| `model.lm_head` | `model.lm_head` |

`set_model()` đi qua `resolve_submodules()` nên chạy được cả hai. Nhân tiện sửa
`model.lm_head.requires_grad = True` — gán `requires_grad` lên một `nn.Module`
chỉ tạo ra một thuộc tính vô nghĩa, không đóng băng gì cả; phải duyệt từng
parameter.

### Vì sao phải lọc flag

`transformers.TrainingArguments` đổi giữa 4.x và 5.x: `warmup_ratio` gộp vào
`warmup_steps` (float trong [0,1) là tỷ lệ), `logging_dir` bỏ hẳn.
`HfArgumentParser` gặp flag lạ là abort **trước khi** load model:

```
ValueError: Some specified arguments are not used by the HfArgumentParser:
['--warmup_ratio', '0.03', '--logging_dir', './output/colab_run/tensorboard/...']
```

`tools/filter_train_args.py` đối chiếu với dataclass thật của bản đang cài, đổi
tên khi có ánh xạ và bỏ flag không nhận, có in cảnh báo ra stderr chứ không âm
thầm.

### Vì sao bỏ monkeypatch attention

`hunyuan_vl` chỉ lên transformers từ **5.13.0**, và lên với tên khác:
`xdrope_section` → `mrope_section`, `apply_rotary_pos_emb_xdrope` →
`apply_multimodal_rotary_pos_emb`, `HunYuanVLAttention` →
`HunYuanVLDenseV1Attention`. Không bản phát hành nào từng có cái tên repo import,
nên `transformers>=4.57.0` trong `requirements.txt` không wheel nào thoả được —
code train viết cho một nhánh pre-merge.

May là bản patch giờ thừa: `HunYuanVLDenseV1Attention` đã đi qua
`ALL_ATTENTION_FUNCTIONS` (nên `flash_attention_2` có hiệu lực) và
`HunYuanVLModel.forward` đã nhận diện packed position_ids
(`ndim == 3 and shape[0] == num_mrope_axes + 1`) để dựng mask — đúng những gì
patch sinh ra để làm.

## 6. Ghi chú về quy mô

Bộ này có 1.007 trang train. Recipe của Tencent dùng ~1M pack cho pretrain và
14,7k pack cho domain finetune, tức là ta ít hơn khoảng hai bậc. Kỳ vọng thực tế
là **thích nghi domain** (đúng bố cục biểu mẫu VN, đúng quy ước Markdown trong
`ocr_prompt.md`), không phải nâng năng lực OCR tổng quát. Nếu loss train tụt
nhanh mà chất lượng trên validation không lên, đó là overfit — hạ `EPOCHS`
xuống 2–3 và/hoặc đóng băng vision tower trước khi nghĩ tới việc đổi LR.
