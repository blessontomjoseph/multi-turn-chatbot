# rouge
# bleau
# perplexity
import pandas as pd
import numpy as np
import torch
from tqdm.autonotebook import tqdm
from datasets import load_metric
from transformers import AutoModelForCausalLM
from utils import tokenizer
from config import InfConfig
rouge_metric=load_metric("rouge")
bleu_metric= load_metric('sacrebleu')


def return_index(input_id,token):
    store=[]
    for index,text in enumerate(input_id):
        if text==token:
            store.append(index)
    return store

def return_speaker_index(input_ids):
    speaker_1=[]
    speaker_2=[]
    for input_id in tqdm(input_ids,desc="speaker_index_extracting"):
        speaker_1.append(return_index(input_id,tokenizer.vocab['<|speaker-1|>']))
        speaker_2.append(return_index(input_id,tokenizer.vocab['<|speaker-2|>']))
    return speaker_1,speaker_2
        

 
def get_metrics(metric_name,model_chkpt,input_ids,token_type_ids,device):
    """
    metric_name:
    "rouge"-for rouge_metric
    "bleau"-for bleau_metric
    
    """
    eval_metric = load_metric(metric_name)
    model=AutoModelForCausalLM.from_pretrained(model_chkpt).to(device)
    speaker_1_indices,speaker_2_indices=return_speaker_index(input_ids)
    for ids,input_id in tqdm(enumerate(input_ids),desc='running',total=len(input_ids)):
        sp2=speaker_1_indices[ids]
        sp1=speaker_2_indices[ids][:len(sp2)]
        sp1.append(-1)
        label_batch=[]
        pred_batch=[]
        for step in range(len(sp2)):
            current_input=torch.LongTensor(input_id[:sp2[step]+1]).unsqueeze(0).to(device)
            current_token_type_id=torch.LongTensor(token_type_ids[ids][:sp2[step]+1]).unsqueeze(0).to(device)
            label_batch.append([tokenizer.decode(input_id[sp2[step]+1:sp1[step+1]])])

            with torch.no_grad():
                output= model.generate(input_ids=current_input,
                                        token_type_ids=current_token_type_id,
                                        pad_token_id=tokenizer.vocab["[pad]"], 
                                        do_sample=True,
                                        top_p=InfConfig.top_p, 
                                        max_new_tokens=InfConfig.max_length, 
                                        min_new_tokens=InfConfig.min_length,
                                        output_hidden_states=True, 
                                        output_scores=True, 
                                        return_dict_in_generate=True).sequences
                
            output=output[0][len(current_input[0]):]
            output=tokenizer.decode(output,skip_special_tokens=True)
            output=output.replace('<|speaker-1|>','').replace('<|speaker-2|>','')
            pred_batch.append([output])

            
        eval_metric.add_batch(predictions=pred_batch,references=label_batch)
    
    if metric_name=="rouge":
        score=eval_metric.compute()
        score={i:score[i].mid.fmeasure for i in score.keys()}
        return pd.DataFrame.from_dict(score, orient="index", columns=["Value"])
    elif  metric_name=='sacrebleu':
        score=eval_metric.compute(smooth_method="floor", smooth_value=0)
        score["precisions"] = [np.round(p, 2) for p in score["precisions"]]
        return pd.DataFrame.from_dict(score, orient="index", columns=["Value"])
            
