# chat demo
from utils import model,tokenizer,device
from itertools import chain
import torch
from config import InfConfig
from transformers import AutoModelForCausalLM


bos_id = tokenizer.vocab['<|startoftext|>']
eos_id = tokenizer.vocab['<|endoftext|>']
speaker_1_id = tokenizer.vocab['<|speaker-1|>']
speaker_2_id = tokenizer.vocab['<|speaker-2|>']
mask = tokenizer.vocab['[mask]']

def chat(model):
    query_history = []
    while True:
        utterance = input('You: ')
        if utterance == "close_chat": #any custom prompt to stop chat
            break
        
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utterance))
        input_ids = [speaker_1_id] + input_ids
        query_history.append(input_ids)
        
        if len(query_history) > InfConfig.max_turns:
            num_exceeded = len(query_history) -  InfConfig.max_turns + 1
            query_history = query_history[num_exceeded:]
            

        input_ids = [bos_id] + list(chain.from_iterable(query_history)) + [speaker_2_id]
        start_sp_id = query_history[0][0]
        next_sp_id = speaker_1_id if start_sp_id == speaker_2_id else speaker_2_id
        token_type_ids = [[start_sp_id] * len(turn) if h % 2 == 0 else [next_sp_id] * len(turn) for h, turn in enumerate(query_history)]
        token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [speaker_2_id]
        input_len = len(input_ids)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(device)
        
        output_ids = model.generate(input_ids=input_ids, 
                                    token_type_ids=token_type_ids, 
                                    pad_token_id=tokenizer.vocab["[pad]"], 
                                    **InfConfig.generation_args
                                   ).sequences
        
        output_ids = output_ids[0].tolist()[input_len:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        response.replace("<|speaker-1|>",'').replace("<|speaker-2|>",'')
        print(f'Bot: {response}')
        query_history.append([speaker_2_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response)))    


if __name__=="__main__":
    inf_checkpoint="theothertom/gpt_neo_whatsapp" # any  model chrckpoint to infer on
    model=AutoModelForCausalLM.from_pretrained(inf_checkpoint).to(InfConfig.device)
    chat(model)