from transformers import LlamaTokenizer,AutoTokenizer
from datasets import Dataset,load_dataset
from preprocess_whatsapp import return_chat_data as human_data_preprocess
from preprocess_daily_dialog import return_chat_data as daily_dialog_preprocess
from datasets import load_from_disk

llama_tokenizer=LlamaTokenizer.from_pretrained("tokenizers/llama_3b_tokenizer")
gpt_neo_tokenizer=AutoTokenizer.from_pretrained("tokenizers/gpt_neo_125m_tokenizer")

if __name__=="__main__":
    human_chat_path="raw_data/human_chat.txt"  #from local
    daily_dailog_data=load_dataset('daily_dialog') #from HF
    
    human_chat_llama=human_data_preprocess(human_chat_path,0.8,0.1,llama_tokenizer)
    human_chat_llama.save_to_disk('processed_data/human_chat_llama_3b')
    human_chat_neo=human_data_preprocess(human_chat_path,0.8,0.1,gpt_neo_tokenizer)
    human_chat_neo.save_to_disk('processed_data/human_chat_neo_125')
    
    
    daily_dailog_llama=daily_dialog_preprocess(daily_dailog_data,llama_tokenizer)
    daily_dailog_llama.save_to_disk('processed_data/daily_dialog_llama_3b')
    daily_dailog_neo=daily_dialog_preprocess(daily_dailog_data,gpt_neo_tokenizer)
    daily_dailog_neo.save_to_disk('processed_data/daily_dialog_neo_125')
    
