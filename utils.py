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


class MyCollator:
    def __init__(self, pad_token_id, max_length):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch):
        input_ids_batch = [torch.tensor(item['input_ids'],dtype=torch.long) for item in batch]
        token_type_ids_batch = [torch.tensor(item['token_type_ids'],dtype=torch.long) for item in batch]
        labels_batch = [torch.tensor(item['labels'],dtype=torch.long) for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.pad_token_id)
        input_ids_padded = input_ids_padded[:, :self.max_length]  # Trim to max_length

        token_type_ids_padded = torch.nn.utils.rnn.pad_sequence(token_type_ids_batch, batch_first=True, padding_value=self.pad_token_id)
        token_type_ids_padded = token_type_ids_padded[:, :self.max_length]  # Trim to max_length

        labels_padded = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = labels_padded[:, :self.max_length]  # Trim to max_length

        return {
            'input_ids': input_ids_padded,
            'token_type_ids': token_type_ids_padded,
            'labels': labels_padded
        }
    
