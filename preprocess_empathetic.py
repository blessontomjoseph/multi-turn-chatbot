from utils import tokenizer
from datasets import load_dataset
empethetic_dialogues=load_dataset("empathetic_dialogues")

from tqdm import tqdm
def return_tokens(data):
    input_ids=[]
    token_type_ids=[]
    labels=[]
    for idx in  tqdm(range(len(data['prompt']))):
        sp_1=["<|startoftext|>","<|speaker-1|>"]+tokenizer.tokenize(data['prompt'][idx])
        sp_2=["<|speaker-2|>"]+tokenizer.tokenize(data['utterance'][idx])+["<|endoftext|>"]
        input_ids.append(tokenizer.convert_tokens_to_ids(sp_1+sp_2))
        token_type_ids.append(tokenizer.convert_tokens_to_ids(["<|speaker-1|>"]*len(sp_1)+["<|speaker-2|>"]*len(sp_2)))
        labels.append(tokenizer.convert_tokens_to_ids(["[mask]"]*len(sp_1)+sp_2))
    return input_ids,token_type_ids,labels

emp_input_ids,emp_token_type_ids,emp_labels=return_tokens(empethetic_dialogues['train'])   
emp_input_ids,emp_token_type_ids,emp_labels=return_tokens(empethetic_dialogues['test'])
emp_input_ids,emp_token_type_ids,emp_labels=return_tokens(empethetic_dialogues['validation'])   