from tqdm.autonotebook import tqdm
from itertools import chain
from datasets import Dataset,DatasetDict


def speaker_dialogs(data,tokenizer):
    vocab=tokenizer.get_vocab()
    token_ids = [] 
    for dialogue in tqdm(data):
        dialogue_ids = []
        for ids, utterance in enumerate(dialogue):
            tokens = tokenizer.tokenize(utterance)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            dialogue_ids.append(ids)
        token_ids.append(dialogue_ids)

    dialogues_with_speaker_ids = []
    for dialogue in tqdm(token_ids):
        utterances_with_speaker_ids = []
        for i, utterance in enumerate(dialogue):
            if i % 2 == 0:
                utterances_with_speaker_ids.append([vocab['<|speaker-1|>']] + utterance)
            else:
                utterances_with_speaker_ids.append([vocab['<|speaker-2|>']] + utterance)
        dialogues_with_speaker_ids.append(utterances_with_speaker_ids)
    return dialogues_with_speaker_ids



def return_tokens(dialogues_with_speaker_ids,tokenizer):
    vocab=tokenizer.get_vocab()
    input_ids=[]
    for conv in dialogues_with_speaker_ids:
        conv_id=[]
        for ids,turn in enumerate(conv):
            if ids==0:
                conv_id.append([vocab['<|startoftext|>']]+turn)
            elif ids==len(conv)-1:
                conv_id.append(turn+[vocab['<|endoftext|>']])
            else:
                conv_id.append(turn)
                
        input_ids.append(conv_id)
                
    
    token_type_ids=[]
    for conv in input_ids:
        conv_id=[]
        for ids,turn in enumerate(conv):
            if ids%2==0:
                conv_id.append([vocab['<|speaker-1|>'] for _ in range(len(turn))])
            else:
                conv_id.append([vocab['<|speaker-2|>'] for _ in range(len(turn))])
        token_type_ids.append(conv_id)
    
    labels=[]
    for conv in input_ids:
        conv_id=[]
        for ids,turn in enumerate(conv):
            if ids%2==0:
                conv_id.append([vocab["[mask]"] for _ in range(len(turn))])
            else:
                conv_id.append(turn)
                
        labels.append(conv_id)
    return input_ids,token_type_ids,labels


def text_to_tokens(data,tokenizer):
    dialogs_with_speaker=speaker_dialogs(data,tokenizer)
    input_ids,token_type_ids,labels=return_tokens(dialogs_with_speaker,tokenizer)                    
    
    input_ids=[list(chain(*input_ids[i])) for i in range(len(input_ids))]
    token_type_ids=[list(chain(*token_type_ids[i])) for i in range(len(token_type_ids))]
    labels=[list(chain(*labels[i])) for i in range(len(labels))]
    return input_ids,token_type_ids,labels

def return_chat_data(dialog,tokenizer):
    train_data=dialog['train']['dialog']
    val_data=dialog['validation']['dialog']
    test_data=dialog['test']['dialog']

    train_input_ids,train_token_type_ids,train_labels=text_to_tokens(train_data,tokenizer) 
    val_input_ids,val_token_type_ids,val_labels=text_to_tokens(val_data,tokenizer) 
    test_input_ids,test_token_type_ids,test_labels=text_to_tokens(test_data,tokenizer) 

    # preparing data to datasetdict format
    train_dataset = Dataset.from_dict({"input_ids":train_input_ids,
                            'token_type_ids':train_token_type_ids,
                            'labels':train_labels})

    validation_dataset = Dataset.from_dict({"input_ids":val_input_ids,
                                'token_type_ids':val_token_type_ids,
                                'labels':val_labels})

    test_dataset = Dataset.from_dict({"input_ids":test_input_ids,
                        'token_type_ids':test_token_type_ids,
                        'labels':test_labels})

    chat_dataset_daily_dialogue = DatasetDict({'train': train_dataset,
                            'validation': validation_dataset,
                            'test': test_dataset})
    return chat_dataset_daily_dialogue