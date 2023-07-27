import re
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from tqdm.autonotebook import tqdm
from datasets import Dataset, DatasetDict,load_from_disk
from transformers import AutoTokenizer,LlamaTokenizer

# Human Conversation training data taken from : https://www.kaggle.com/datasets/projjal1/human-conversation-training-data

def process(data_url:str):
    data = pd.read_csv(data_url, delimiter="\t",header=None)
    data.columns=['chats']
    chats=data['chats'].tolist()

    pattern = r"(?=.*\bhi\b)(?=.*\bHuman 1:).*"
    conv_ids=[idx for idx,turn in enumerate(chats) if re.match(pattern, turn, re.IGNORECASE)!=None]
    dialogues=[chats[conv_ids[idx]:conv_ids[idx+1]] for idx in range(len(conv_ids)-1)]
    dialogues=[['<|startoftext|>']+dialog+['<|endoftext|>'] for dialog in dialogues]
    dialogues=[list(map(lambda x: x.replace('Human 1:','<|speaker-1|>').replace('Human 2:','<|speaker-2|>'),turn)) for turn in dialogues]
    return dialogues

def input_ids_map(dialog, tokenizer):
    return list(map(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)), dialog))

def token_type_ids_map(input_id, tokenizer):
    vocab=tokenizer.get_vocab()
        
    token_type_id = [[vocab['<|speaker-1|>']] * len(turn) if id % 2 == 0 else [vocab['<|speaker-2|>']] * len(turn) for id, turn in enumerate(input_id[1:-1])]
    token_type_id.insert(0, [vocab['<|speaker-1|>']])
    token_type_id.append([vocab['<|speaker-2|>']])
    return token_type_id

def labels_map(input_id, tokenizer):
    vocab=tokenizer.get_vocab()
        
    label = [[vocab['[mask]']] * len(turn) if id % 2 == 0 else turn for id, turn in enumerate(input_id[1:-1])]
    label.insert(0, [vocab['[mask]']])
    label.append([vocab['<|endoftext|>']])
    return label

def return_tokens(dialogues, tokenizer):
    input_ids = list(map(lambda dialog: input_ids_map(dialog, tokenizer), tqdm(dialogues)))
    token_type_ids = list(map(lambda input_id: token_type_ids_map(input_id, tokenizer), tqdm(input_ids)))
    labels = list(map(lambda input_id: labels_map(input_id, tokenizer), tqdm(input_ids)))

    input_ids = [list(chain(*input_id)) for input_id in input_ids]
    token_type_ids = [list(chain(*token_type_id)) for token_type_id in token_type_ids]
    labels = [list(chain(*label)) for label in labels]
    return input_ids, token_type_ids, labels


def dataset_dict_format(input_ids,token_type_ids,labels,train_split,val_split):

    train=int(np.round(len(input_ids)*(train_split)))
    val=int(np.round(len(input_ids)*(val_split)))+train


    train_dataset = Dataset.from_dict({"input_ids":input_ids[:train],
                            'token_type_ids':token_type_ids[:train],
                            'labels':labels[:train]})

    validation_dataset = Dataset.from_dict({"input_ids":input_ids[train:val],
                                'token_type_ids':token_type_ids[train:val],
                                'labels':labels[train:val]})

    test_dataset = Dataset.from_dict({"input_ids":input_ids[val:],
                        'token_type_ids':token_type_ids[val:],
                        'labels':labels[val:]})

    chat_dataset_whatsapp = DatasetDict({'train': train_dataset,
                            'validation': validation_dataset,
                            'test': test_dataset})
    return chat_dataset_whatsapp


def return_chat_data(data_url,train_split,val_split,tokenizer):
    """from data path and split sizes returns a dataset with corresponding splits
        train_size=train_split
        val_size=val_split
        test_size=1-(train_size+val_size)
        
    Args:
        data_url (str): data path
        train_split (float): train split in [0,1]
        val_split (float): vlidation_split in [0,1]

    Returns:
        DatasetDict object
    """
    dialogues=process(data_url)
    input_ids,token_type_ids,labels=return_tokens(dialogues,tokenizer)
    chat_data=dataset_dict_format(input_ids,token_type_ids,labels, train_split,val_split)
    return chat_data

