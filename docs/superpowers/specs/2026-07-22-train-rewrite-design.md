# train.py 重写 — Unsloth + QLoRA + PEFT 微调 Gemma 4 E4B

**Date**: 2026-07-22
**Status**: 设计完成

---

## 1. 背景

`train/train.py` 有 6 个关键问题导致训练无效：

| # | 问题 | 影响 |
|---|------|------|
| 1 | 无 Gemma chat template（`<start_of_turn>`） | 训练格式与推理不一致，模型学错格式 |
| 2 | `DataCollatorForSeq2Seq` 用于 decoder-only 模型 | 错误的填充策略 |
| 3 | 无 loss masking（prompt 部分也算 loss） | 浪费算力学习 prompt 模板 |
| 4 | `MAX_SEQ_LENGTH=2048` | 截断，prompt + JSON 输出可达 4000+ tokens |
| 5 | prompt 格式不匹配 `extract.txt` | 与推理时给模型的指令完全不同 |
| 6 | {triples} 用 Python repr 而非 JSON | 输出格式错误（单引号 vs 双引号） |

## 2. 设计目标

重写 `train/train.py`，使：

1. 训练 prompt 与推理时 `kg_extract.py` 的 `build_gemma_prompt()` 100% 一致
2. Loss 只计算 assistant 部分（model response），不计算 system/user 部分
3. 输出为紧凑 JSON 数组（匹配 `extract.txt` 的输出格式要求）
4. QLoRA 4bit 加载 + LoRA adapter，适合 12GB VRAM
5. 训练后可导出 LoRA 权重，可选 merge 回模型再导出 GGUF

## 3. 改动文件

只改一个文件：`train/train.py`（完全重写，约 150 行）

不动：
- `train/train_data/triples.jsonl` — 训练数据文件，格式不变
- `prompts/extract.txt` — 只读引用，不改

## 4. Chat Template 构建

### 4.1 推理时的格式（必须匹配）

来自 `scripts/kg_extract.py` 的 `build_gemma_prompt()`：

```
<start_of_turn>system
你是资深黄河流域水库调度专家。只输出JSON数组，不输出任何解释。<end_of_turn>
<start_of_turn>user
{extract.txt 模板填入 text_chunk}<end_of_turn>
<start_of_turn>model
[{"subject":"...","relation":"...","object":"...","context":"...","confidence":0.95}]<end_of_turn>
```

### 4.2 训练时的构建方式

```python
GEMMA_SYSTEM = "你是资深黄河流域水库调度专家。只输出JSON数组，不输出任何解释。"

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TEXT_PROMPT_TEMPLATE = open(PROMPTS_DIR / "extract.txt", encoding="utf-8").read()

def format_training_example(text: str, triples: list) -> str:
    """构建与推理时完全一致的训练样本"""
    user_content = TEXT_PROMPT_TEMPLATE.format(text_chunk=text)
    assistant_content = json.dumps(triples, ensure_ascii=False, separators=(",", ":"))

    return (
        f"<start_of_turn>system\n{GEMMA_SYSTEM}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant_content}<end_of_turn>"
    )
```

### 4.3 Loss Masking 策略

- 使用 TRL 的 `DataCollatorForCompletionOnlyLM`
- `response_template = "<start_of_turn>model\n"`，只计算该标记之后的 loss
- 前导空格/特殊字符由 tokenizer 正确编码

## 5. QLoRA + 训练参数

### 5.1 模型加载

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-4-4b-it",
    max_seq_length=8192,
    load_in_4bit=True,
)
```

### 5.2 LoRA 配置

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # 领域特定任务用更高 rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### 5.3 训练参数

| 参数 | 值 | 原因 |
|------|-----|------|
| per_device_batch_size | 1 | 4B + 8192 ctx + 4bit, 约 8GB VRAM |
| gradient_accumulation_steps | 8 | 有效 batch=8 |
| num_epochs | 5 | 9 条小数据集需更多 epoch |
| learning_rate | 2e-4 | LoRA 标准 |
| lr_scheduler | cosine | |
| optim | adamw_8bit | |
| bf16 | True | Ampere+ GPU 优先，否则 fp16 |
| warmup_ratio | 0.05 | |
| weight_decay | 0.01 | |
| logging_steps | 1 | 小数据集每个 step 都 log |
| save_strategy | epoch | |

### 5.4 Trainer

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForCompletionOnlyLM(
        response_template="<start_of_turn>model\n",
        tokenizer=tokenizer,
    ),
    formatting_func=format_training_example,
    args=TrainingArguments(...),
)
```

**不使用 `dataset_text_field`** — 改用 `formatting_func`，让 SFTTrainer 内部完成 tokenization 和 loss masking。

### 5.5 数据路径

```python
DATA_PATH = "train_data/triples.jsonl"  # 相对 train.py 的路径
OUTPUT_DIR = "./output/gemma4-hydro-lora"
```

## 6. 输出

### 6.1 保存

```python
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

输出到 `train/output/gemma4-hydro-lora/`：
- adapter_config.json
- adapter_model.safetensors
- tokenizer files

### 6.2 可选：导出 GGUF（供 llama-cpp 推理）

```python
# 1. Merge LoRA back to base model
model.save_pretrained_merged(OUTPUT_DIR + "_merged", tokenizer, save_method="merged_16bit")

# 2. Convert to GGUF using llama.cpp convert_hf_to_gguf.py
# (手动在另一个环境执行)
```

## 7. 验证

```bash
conda activate train
cd d:/work/knowlegeextract/train
python train.py
```

验证点：
1. 模型成功从 HF 加载 google/gemma-4-4b-it（4bit QLoRA）
2. 数据集正确加载（9 条）
3. prompt 格式包含 `<start_of_turn>system/user/model` 标签
4. Loss 只计算 assistant 部分
5. 训练完成，loss 下降
6. LoRA adapter 保存到 output/
7. 加载 adapter 后，模型能对水文文本输出正确 JSON 数组
