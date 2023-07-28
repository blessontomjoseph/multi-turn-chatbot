import utils
from config import TrainConfig
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import preprocess_daily_dialog
import metrics
import os
import pandas as pd
from transformers import AutoModelForCausalLM,AutoTokenizer

hf_user="" #specify hf username for model to be downloaded fr inference or evaluation
hub_repository="" #specify a  hf hub repository name for the model to be uploaded after training

if __name__=="__main__":
    
    model=AutoModelForCausalLM.from_prettrained(TrainConfig.checkpoint).to(TrainConfig.device)
    tokenizer=AutoTokenizer.from_pretrained("tokenizer/gpt_neo_125_tokenizer")
    collator_with_padding=DataCollatorWithPadding(tokenizer=tokenizer)
    dialog=load_dataset('daily_dialog') # loads the  unprocessed data (open it from a file system or load it from hf datasets)
    
    data=preprocess_daily_dialog.return_chat_data(dialog) #this does the proprocessing
    utils.train_model(hub_repository,data,model,tokenizer,collator_with_padding) #trains the model
    
    # data for testing
    test_input_ids=data['test']['input_ids']
    test_token_type_ids=data['test']['token_type_ids']

    model_chkpt=os.path.join(hf_user,hub_repository)
    rouge_score=metrics.get_metrics('rouge',model_chkpt,test_input_ids,test_token_type_ids,'cpu')
    bleu_score=metrics.get_metrics('bleu',model_chkpt,test_input_ids,test_token_type_ids,'cpu')
    pd.to_csv(rouge_score,"rouge_score.csv")
    pd.to_csv(bleu_score,"bleu_score.csv")
    
    

