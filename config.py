import torch

class TrainConfig:
    checkpoint="EleutherAI/gpt-neo-125m"  # paste a checkpoint
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_batch_size_lightning=1
    eval_bathc_size_lightning=1
    
    general_training_args={'output_dir':'./',
                        'overwrite_output_dir':True, 
                        'push_to_hub':True,
                                
                        'num_train_epochs':10,
                        'warmup_steps':10,
                        'weight_decay':0.1, 
                        'logging_steps':10,
                        'evaluation_strategy':'steps',
                        'eval_steps':0.1,
                        'save_steps':0.1,
                       
                        'optim':'adamw_torch', 
                        'save_strategy':'steps', 
                        'save_total_limit':1, 
                        'logging_dir':'/kagle/working/logs' ,
                        'report_to':'wandb',  

                        'per_device_train_batch_size':1, 
                        'per_device_eval_batch_size':1,
                        'gradient_accumulation_steps':8}

    
    
class InfConfig:
    max_turns = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generation_args= {"do_sample":True, 
                        "top_p":0.8, 
                        "max_new_tokens":8,
                        "min_new_tokens":2,
                        "output_hidden_states":True, 
                        "output_scores":True, 
                        "return_dict_in_generate":True}
    
    
    
    
