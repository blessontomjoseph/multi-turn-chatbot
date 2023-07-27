from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from huggingface_hub import login
from config import TrainConfig


def add_special_tokens(tokenizer):
    special_tokens={"bos_token":"<|startoftext|>",
                    "eos_token":"<|endoftext|>",
                    "mask_token":"[mask]",
                    "pad_token":"[pad]",}

    other_tokens=["<|speaker-1|>","<|speaker-2|>"]
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(other_tokens)
    return tokenizer


def train_model(hub_repo_name:str,data,model,tokenizer,collator):
    login()
    trainer=Trainer(model=model,
        args=TrainingArguments(push_to_hub_model_id=hub_repo_name,**TrainConfig.general_training_args),
        tokenizer=tokenizer,
        data_collator= collator,
        train_dataset=data['train'],
        eval_dataset =data['validation'])
    trainer.train()
