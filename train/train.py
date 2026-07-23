# =====================================================
# Gemma 4 E4B IT — 水文三元组抽取微调
# Unsloth + QLoRA + PEFT (8GB VRAM)
# =====================================================

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# DataCollatorForCompletionOnlyLM: TRL >= 0.9.0 原生支持，低版本手动实现
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorForCompletionOnlyLM:
        """TRL < 0.9.0 兼容实现: 只计算 response 部分的 loss"""
        response_template: str
        tokenizer: Any

        def __post_init__(self):
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )

        def __call__(self, examples: List[Union[str, Dict]]) -> Dict:
            # 提取文本
            texts = []
            for ex in examples:
                if isinstance(ex, dict):
                    texts.append(ex.get("text", ex.get("input", str(ex))))
                else:
                    texts.append(str(ex))

            # Tokenize
            batch = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )

            # Loss masking: response 之前的所有 token 设为 -100
            labels = batch["input_ids"].clone()

            for i in range(labels.size(0)):
                # 在 input_ids 中查找 response_token_ids 的位置
                seq = labels[i].tolist()
                resp_start = -1
                for j in range(len(seq) - len(self.response_token_ids) + 1):
                    if seq[j:j + len(self.response_token_ids)] == self.response_token_ids:
                        resp_start = j + len(self.response_token_ids)
                        break

                if resp_start > 0:
                    labels[i, :resp_start] = -100  # prompt 部分不算 loss
                # 如果没找到 response_template，整个序列都不算 loss（安全兜底）
                elif resp_start == -1:
                    labels[i, :] = -100

            batch["labels"] = labels
            return batch

# ============================
# 1. 参数配置（8GB VRAM）
# ============================

MODEL_NAME = "google/gemma-4-4b-it"
DATA_PATH = "train_data/triples.jsonl"         # 相对 train/ 目录
OUTPUT_DIR = "./output/gemma4-hydro-lora"
MAX_SEQ_LENGTH = 4096

GEMMA_SYSTEM = "你是资深黄河流域水库调度专家。只输出JSON数组，不输出任何解释。"

# 从项目 prompts 目录读取模板（与推理时完全一致）
BASE_DIR = Path(__file__).parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
TEXT_PROMPT_TEMPLATE = open(PROMPTS_DIR / "extract.txt", encoding="utf-8").read()


def format_training_example(example: dict) -> str:
    """
    构建与 kg_extract.py build_gemma_prompt() 完全一致的训练样本。
    格式:
      <start_of_turn>system\n{GEMMA_SYSTEM}<end_of_turn>\n
      <start_of_turn>user\n{extract.txt 模板填入 text}<end_of_turn>\n
      <start_of_turn>model\n{紧凑 JSON}<end_of_turn>
    """
    text = example["text"]
    triples = example["triples"]

    # User content: 使用与推理相同的 extract.txt 模板
    user_content = TEXT_PROMPT_TEMPLATE.format(text_chunk=text)

    # Assistant content: 紧凑 JSON 数组（匹配提取输出格式）
    assistant_content = json.dumps(triples, ensure_ascii=False, separators=(",", ":"))

    # 手动构建 <start_of_turn> prompt（与 build_gemma_prompt 一致）
    prompt = (
        f"<start_of_turn>system\n{GEMMA_SYSTEM}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant_content}<end_of_turn>"
    )
    return prompt


# ============================
# 2. 加载模型（4-bit QLoRA）
# ============================

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    # 8GB 显存优化
    dtype=torch.float16,
)
print("Model loaded.")

# ============================
# 3. 添加 LoRA
# ============================

model = FastLanguageModel.get_peft_model(
    model,
    r=8,                     # 8GB VRAM 适配
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",  # 长上下文显存优化
    random_state=42,
)
print("LoRA attached.")

# ============================
# 4. 加载数据
# ============================

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# ============================
# 5. 验证 prompt 格式（训练前确认一次）
# ============================

sample = dataset[0]
sample_prompt = format_training_example(sample)
print(f"\n--- Prompt preview (first 300 chars) ---")
print(sample_prompt[:300])
print(f"... (total {len(sample_prompt)} chars)")
# 确认包含所有关键标签
assert "<start_of_turn>system\n" in sample_prompt, "Missing system tag"
assert "<start_of_turn>user\n" in sample_prompt, "Missing user tag"
assert "<start_of_turn>model\n" in sample_prompt, "Missing model tag"
assert "<end_of_turn>" in sample_prompt, "Missing end_of_turn tag"
assert GEMMA_SYSTEM in sample_prompt, "Missing system message"
print("Prompt format: OK")

# ============================
# 6. 配置 Trainer
# ============================

# Loss masking: 只计算 <start_of_turn>model\n 之后的部分
response_template = "<start_of_turn>model\n"
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# 确保 response_template token 被正确识别
# 检查 tokenizer 对 response_template 的编码
response_tokens = tokenizer.encode(response_template, add_special_tokens=False)
print(f"Response template tokens: {response_tokens} (first 3: {response_tokens[:3]})")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=data_collator,
    formatting_func=format_training_example,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        # 8GB VRAM 适配
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,      # 有效 batch=8
        num_train_epochs=5,                 # 小数据集多跑几轮
        learning_rate=2e-4,
        fp16=True,                          # 省显存
        logging_steps=1,                    # 每步记录（9条数据只有几步）
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    ),
)

# ============================
# 7. 开始训练
# ============================

print("\nStart training...")
trainer_stats = trainer.train()
print(f"Training completed: {trainer_stats}")

# ============================
# 8. 保存 LoRA Adapter
# ============================

print(f"\nSaving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 同时保存训练配置，供后续合并/推理使用
config = {
    "model_name": MODEL_NAME,
    "max_seq_length": MAX_SEQ_LENGTH,
    "lora_r": 8,
    "lora_alpha": 16,
    "system_prompt": GEMMA_SYSTEM,
    "prompt_template": "prompts/extract.txt",
}
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"Adapter saved.")
print(f"Done! Output: {OUTPUT_DIR}")
print(f"Next: merge adapter -> convert to GGUF -> replace models/gemma-4-E4B-it-Q5_K_M.gguf")
