# coding=utf-8

import logging
import math, os, sys
from dataclasses import dataclass

import datasets as ds
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


@dataclass
class Args(TrainingArguments):
    model_name_or_path: str = "bert-base-cased"
    train_file: str = "./data/sample.txt"
    validation_file: str = "./data/dev.txt"
    tokenizer_name: str = "./tokenizer/vocab.txt"
    do_train: bool = True
    do_eval: bool = True
    output_dir: str = "./output"
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


def main():

    args = Args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    data_files = {}
    data_files["train"] = args.train_file
    data_files["validation"] = args.validation_file
    extension = "text"

    datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir)

    config_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": "main",
        "use_auth_token": None,
    }
    config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs,\
        model_max_length=args.max_seq_length, do_lower_case=True)

    # Model
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
        revision="main",
        use_auth_token=None,
    )
    model.resize_token_embeddings(len(tokenizer))

    # 텍스트 토크나이징
    if args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = "max_length" if args.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache
    )

    # 랜덤 마스킹
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"] if args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if args.do_train:
        model_path = (
            args.model_name_or_path
            if (args.model_name_or_path is not None and os.path.isdir(args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model() 

        output_train_file = os.path.join(args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
            trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(args.output_dir, "eval_results_mlm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


if __name__ == "__main__":
    main()
