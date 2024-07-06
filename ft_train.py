import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
import torch
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType, get_peft_model

@dataclass
class ScriptArguments:
    """ 训练参数相关参数
    """
    mode_path : Optional[str] = field(default=" ", metadata={"help": "SFT train, the base model path"})
    dataset_dir_or_path : Optional[str] = field(default=" ", metadata={"help": "train dataset dir or path"})
    resume : Optional[bool] = field(default=False, metadata={"help": "use PyTorch 2.0 to compile the model to be faster"})

    # transformer args---------------- 
    seed : Optional[int] = field(default=42, metadata={"help": "seed"})
    output_dir :  Optional[str] = field(default=".", metadata={"help": "train output dir"})
    ds_config_json : Optional[str] = field(default="")
    ds_type : Optional[str] = field(default="fp16")
    mbs : Optional[int] = field(default=32, metadata={"help": "per_device_train_batch_size"})
    gas : Optional[int] = field(default=1, metadata={"help": "gradient_accumulation_steps"})
    nums_epochs : Optional[int] = field(default=1)
    tb_dir : Optional[str] = field(default="")
    logging_steps : Optional[int] = field(default=10)
    grad_clip : Optional[float] = field(default=1.0)
    lr_scheduler_type : Optional[str] = field(default="cosine")
    lr : Optional[float] = field(default=1e-5)
    warmup_ratio : Optional[float] = field(default=0.03)
    save_total_limit: Optional[int] = field(default=5)
    save_steps : Optional[int] = field(default=500)

    # lora args ------
    use_lora : Optional[bool] = field(default=False, metadata={"help": "use_lora?"})
    lora_r : Optional[int] = field(default=8, metadata={"help": "lora r"})
    lora_alpha : Optional[int] = field(default=32, metadata={"help": "lora alpha"})
    lora_dropout : Optional[float] = field(default=0.05, metadata={"help": "lora alpha"})

def load_train_dataset(
    data_path: str,
    tokenizer,
    max_length=512,
    sanity_check: bool = False,
    num_proc=24,
    ) -> datasets.Dataset:
    dataset = load_dataset("json", data_files=data_path, split='train', cache_dir="./cache/hf")
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def process_func(example):
        input_ids, attention_mask, labels = [], [], []
        input_text = example["input"].split("User: \n")

        system_text = input_text[0]
        user_text = input_text[1].replace("\nAssistant: \n", "")
        target_text = example['target']

        instruction = tokenizer(f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{target_text}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        if len(input_ids) > max_length:  # 做一个截断
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    dataset_map = dataset.map(
        process_func,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    dataset_map = dataset_map.filter(
        lambda x: len(x["input_ids"]) <= max_length
    )

    dataset_map.set_format(type="torch")

    return dataset_map

def load_model_tokenizer(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, config=config)
    # model.to(device)
    return model, tokenizer, config

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args, = parser.parse_args_into_dataclasses()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, config = load_model_tokenizer(script_args.mode_path, device)
    train_dataset = load_train_dataset(script_args.dataset_dir_or_path, tokenizer, max_length=4096)

    if script_args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False, # 训练模式
            r=script_args.lora_r, # Lora 秩
            lora_alpha=script_args.lora_alpha, # Lora alaph，具体作用参见 Lora 原理
            lora_dropout=script_args.lora_dropout# Dropout 比例
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_args = TrainingArguments(
        seed=script_args.seed,
        output_dir=script_args.output_dir,
        overwrite_output_dir=True,
        deepspeed=script_args.ds_config_json,
        per_device_train_batch_size=script_args.mbs,
        gradient_accumulation_steps=script_args.gas,
        bf16=script_args.ds_type=="bf16",
        fp16=script_args.ds_type=="fp16",
        num_train_epochs=script_args.nums_epochs,
        logging_dir=script_args.tb_dir,
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=script_args.logging_steps,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=script_args.grad_clip,
        lr_scheduler_type=script_args.lr_scheduler_type,
        learning_rate=script_args.lr,
        warmup_ratio=script_args.warmup_ratio,
        save_strategy="steps",
        save_total_limit=script_args.save_total_limit,
        save_steps=script_args.save_steps,
        ddp_timeout=3000,
        logging_first_step=True,
        save_safetensors=False
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest"),
    )

    trainer.train(script_args.resume)

    # 模型保存
    last_model_dir = os.path.join(script_args.output_dir, 'last_ft_model')
    os.makedirs(last_model_dir, exist_ok=True)
    tokenizer.save_pretrained(last_model_dir)
    if script_args.use_lora:
        trainer.model.save_pretrained(last_model_dir)
    else:
        trainer.save_model(output_dir=last_model_dir)

if __name__ == "main":
    main()

