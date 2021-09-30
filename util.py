from dataclasses import dataclass
from transformers import TrainingArguments


@dataclass
class Base(TrainingArguments):
    model_name_or_path: str = "bert-base-cased"
    train_file: str = "./data/sample.txt"
    validation_file: str = "./data/dev.txt"
    tokenizer_name: str = "./tokenizer/vocab.txt"
    do_train: bool = True
    do_eval: bool = True
    output_dir: str = "./model/bert"
    line_by_line: bool = True
    max_seq_length: int = 300
    preprocessing_num_workers: int = 8
    cache_dir: str = "./model_cache"
    fp16: bool = True
    seed: int = 1234
    pad_to_max_length: bool = False
    overwrite_cache: bool = False
    mlm_probability: int = 0.15
    
    logging_steps: int = 100000
    save_steps: int = 100000
    per_gpu_train_batch_size: int = 100
    per_device_train_batch_size: int = 100
    per_gpu_eval_batch_size: int = 100
    per_device_eval_batch_size: int = 100

@dataclass
class BertArgs(Base):
    model_name_or_path: str = "bert-base-cased"
    train_file: str = "./data/sample.txt"
    validation_file: str = "./data/dev.txt"
    tokenizer_name: str = "./tokenizer/wordpiece/vocab.txt"
    do_train: bool = True
    do_eval: bool = True
    output_dir: str = "./model/bert"

@dataclass
class DistilBertArgs(Base):
    model_name_or_path: str = "distilbert-base-cased"
    train_file: str = "./data/sample.txt"
    validation_file: str = "./data/dev.txt"
    tokenizer_name: str = "./tokenizer/wordpiece/vocab.txt"
    do_train: bool = True
    do_eval: bool = True
    output_dir: str = "./model/distil-bert"

@dataclass
class RoberTaArgs(Base):
    model_name_or_path: str = "roberta-base"
    train_file: str = "./data/sample.txt"
    validation_file: str = "./data/dev.txt"
    tokenizer_name: str = "./tokenizer/bbpe/vocab.json"
    do_train: bool = True
    do_eval: bool = True
    output_dir: str = "./model/roberta"