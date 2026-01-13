import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

from datasets import Dataset as HFDataset
import os
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore")

os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

def load_and_preprocess_data(tokenizer, num_samples, max_length, eval_ratio):
    dataset = load_dataset("Flmc/DISC-Med-SFT", split="train[:1%]")

    def format_conversation(example):
        conv = example["conversation"]
        prompt = ""
        for turn in conv:
            if turn["role"] == "user":
                prompt += f"<|im_start|>user\n{turn['content']}<|im_end|>\n"
            elif turn["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{turn['content']}<|im_end|>\n"
        return {"text": prompt.strip()}

    dataset = dataset.map(format_conversation)
    dataset = dataset.remove_columns(["_id", "source", "conversation"])

    dataset = dataset.train_test_split(test_size=eval_ratio, seed=42)
    print(f"训练集样本数：{len(dataset['train'])}，验证集样本数：{len(dataset['test'])}")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False  # Qwen3无需token_type_ids
        )
    # 批量分词（增大batch_size加快处理）
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )
    tokenized_test = dataset["test"].map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )

    # 转换为PyTorch格式
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])
    print(tokenized_train['input_ids'][0])
    print(tokenized_train['attention_mask'][0])
    return tokenized_train, tokenized_test

def setup_model_and_tokenizer():
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./models_cache",  # 模型缓存路径，避免重复下载
        padding_side="right",        # 避免推理时的显存碎片
        torch_dtype=torch.float16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",  # 替换死写的 cuda:0，更灵活
        trust_remote_code=True,
        cache_dir="./models_cache",
        use_cache=False,  # 与梯度检查点兼容
        #gradient_checkpointing=True  # 开启梯度检查点
        attn_implementation="flash_attention_2"
    )

    # 关键步骤：为4位量化模型准备训练（开启梯度）
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

def compute_metrics(eval_pred):
    """修复评估指标计算逻辑（适配语言模型输出）"""
    predictions, labels = eval_pred
    # 语言模型输出是 logits，先取 argmax
    predictions = np.argmax(predictions, axis=-1)
    
    # 过滤 pad_token（避免影响准确率）
    mask = labels != -100  # DataCollator 会将 pad_token 设为 -100
    labels = labels[mask]
    predictions = predictions[mask]
    
    # 计算准确率
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())
    return {'accuracy': accuracy}

# Debug
def print_mem_usage(stage=""):
  curr_mem = torch.cuda.memory_allocated() / 1024**3  # GB
  peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
  print(f"[{stage}] GPU显存：当前 {curr_mem:.2f}GB，峰值 {peak_mem:.2f}GB")

class MemMonitorCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:  # 每10步打印一次
            print_mem_usage(f"训练步数 {state.global_step}")

def main(num_samples=1000):
    training_args = TrainingArguments(
        output_dir="./qwen_training_results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        learning_rate=5e-5,
        save_total_limit=3,
        prediction_loss_only=False,
        remove_unused_columns=False,
        report_to=None,
        fp16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        # 关键：梯度检查点兼容配置
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    print_mem_usage("加载模型前")
    # 加载模型和分词器
    model, tokenizer = setup_model_and_tokenizer()

    print_mem_usage("加载模型后")
    # 加载数据 - 现在返回训练集和验证集
    tokenized_train, tokenized_test = load_and_preprocess_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=2048,
        eval_ratio=0.05
    )
    print_mem_usage("加载数据后")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    print_mem_usage("创建训练器前")
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    print_mem_usage("微调前")

    trainer.add_callback(MemMonitorCallback())

    train_result = trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained()

    # 显示训练结果摘要
    print("\n训练结果摘要:")
    print(f"使用的样本数量: {num_samples}")
    print(f"最终训练损失: {train_result.training_loss:.4f}")
    print(f"总训练步数: {trainer.state.global_step}")
    print(f"使用的显卡: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")


if __name__ == "__main__":
    # 设置 CUDA 环境变量，确保梯度计算正常
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    NUM_SAMPLES = 500  # 设置想要使用的数据条数，None表示使用全部数据
    main(num_samples=NUM_SAMPLES)
