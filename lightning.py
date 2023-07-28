import torch
import utils
import wandb
import lightning as l
import torch.nn as nn
from torch import optim
from config import TrainConfig
import preprocess_daily_dialog
from torch.utils.data import DataLoader
from datasets import load_metric,load_dataset
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoModelForCausalLM,AutoTokenizer

        
wandb_key=""
hf_key=""
wandb.login(key=wandb_key)

class Config:
    checkpoint="theothertom/gpt_neo_extended_retrain"  #hf hub repo id
    train_batch_size=1
    val_batch_size=1
    default_max_out_tokens=50 #  limit output token size if needed
    
class GptNeoTune(l.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.loss_fn=nn.CrossEntropyLoss()
        self.wandb_logger=WandbLogger()
        
        self.train_losses=[]
        self.val_losses=[]
        
        self.train_rouge_1=[]
        self.val_rouge_1=[]
        
        self.train_rouge_2=[]
        self.val_rouge_2=[]
        
        self.train_rouge_L=[]
        self.val_rouge_L=[]

        self.train_bleu=[]
        self.val_bleu=[]
        
#         self.train_logits=[]
#         self.val_logits=[]
#         self.train_labels=[]
#         self.val_labels=[]
        

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-5)
    
    @staticmethod
    def metrics(logits,labels,tokenizer):
        rouge_metric = load_metric("rouge")
        bleu_metric = load_metric("sacrebleu")
        predictions=logits.argmax(dim=-1)
        predictions=torch.tensor(predictions,dtype=torch.long)
        predictions=tokenizer.batch_decode(predictions,skip_special_tokens=True)
        labels=tokenizer.batch_decode(labels,skip_special_tokens=True)
        rouge_output = rouge_metric.compute(predictions=predictions, references=labels)

        for pred, ref in zip(predictions, labels):
            try:
                pred_sentence = " ".join(pred).replace("[PAD]", "").strip()
                ref_sentence = " ".join(ref).replace("[PAD]", "").strip()
                bleu_metric.add(prediction=[pred_sentence], reference=[ref_sentence])
            except:
#                 debugg stuff
                print('p',predictions)
                print('l',labels)

        bleu_score = bleu_metric.compute()["score"]

        return {
                "rouge_1": rouge_output["rouge1"].mid.fmeasure,
                "rouge_2": rouge_output["rouge2"].mid.fmeasure,
                "rouge_L": rouge_output["rougeL"].mid.fmeasure,
                "bleu":bleu_score
                }
    
    def forward(self,batch):
        out=self.model(input_ids=batch['input_ids'],token_type_ids=batch['token_type_ids'])
        return out['logits']
    
    def training_step(self,batch,batch_idx):
        logits=self.forward(batch)
        loss=self.loss_fn(logits.transpose(1,2),batch['labels'])
        self.train_losses.append(loss.item())
        metrics=self.metrics(logits=logits.transpose(1,2),labels=batch['labels'],tokenizer=tokenizer)
        self.train_rouge_1.append(metrics['rouge_1'])
        self.train_rouge_2.append(metrics['rouge_2'])
        self.train_rouge_L.append(metrics['rouge_L'])
        self.train_bleu.append(metrics['bleu'])
        return loss
    
    def on_train_epoch_end(self):
        mean_train_loss=np.mean(self.train_losses)
        mean_train_rouge_1=np.mean(self.train_rouge_1)
        mean_train_rouge_2=np.mean(self.train_rouge_2)
        mean_train_rouge_L=np.mean(self.train_rouge_L)
        mean_train_bleu=np.mean(self.train_bleu)
        
        self.train_losses.clear()
        self.train_rouge_1.clear()
        self.train_rouge_2.clear()
        self.train_rouge_L.clear()
        self.train_bleu.clear()
        
        print("mean_train_loss: ",mean_train_loss)
        self.log("mean_train_loss",mean_train_loss,on_step=False,on_epoch=True,prog_bar=True)
        self.log("train_rouge_1",mean_train_rouge_1,on_step=False,on_epoch=True,prog_bar=True)
        self.log("train_rouge_2",mean_train_rouge_2,on_step=False,on_epoch=True,prog_bar=True)
        self.log("train_rouge_L",mean_train_rouge_L,on_step=False,on_epoch=True,prog_bar=True)
        self.log("train_bleu",mean_train_bleu,on_step=False,on_epoch=True,prog_bar=True)
        
    def validation_step(self,batch,batch_idx):
        logits=self.forward(batch)
        loss=self.loss_fn(logits.transpose(1,2),batch['labels'])
        self.val_losses.append(loss.item())
        metrics=self.metrics(logits=logits.transpose(1,2),labels=batch['labels'],tokenizer=tokenizer)
        self.val_rouge_1.append(metrics['rouge_1'])
        self.val_rouge_2.append(metrics['rouge_2'])
        self.val_rouge_L.append(metrics['rouge_L'])
        self.val_bleu.append(metrics['bleu'])
        return loss
    
    def on_validation_epoch_end(self):
        mean_val_loss=np.mean(self.val_losses)
        mean_val_rouge_1=np.mean(self.val_rouge_1)
        mean_val_rouge_2=np.mean(self.val_rouge_2)
        mean_val_rouge_L=np.mean(self.val_rouge_L)
        mean_val_bleu=np.mean(self.val_bleu)
        
        self.val_losses.clear()
        self.val_rouge_1.clear()
        self.val_rouge_2.clear()
        self.val_rouge_L.clear()
        self.val_bleu.clear()
        
        print("mean_val_loss: ",mean_val_loss)
        self.log("mean_val_loss",mean_val_loss,on_step=False,on_epoch=True,prog_bar=True)
        self.log("val_rouge_1",mean_val_rouge_1,on_step=False,on_epoch=True,prog_bar=True)
        self.log("val_rouge_2",mean_val_rouge_2,on_step=False,on_epoch=True,prog_bar=True)
        self.log("val_rouge_L",mean_val_rouge_L,on_step=False,on_epoch=True,prog_bar=True)
        self.log("val_bleu",mean_val_bleu,on_step=False,on_epoch=True,prog_bar=True)    

        

        
base_model=AutoModelForCausalLM.from_pretrained(TrainConfig.checkpoint)    
tokenizer=AutoTokenizer.from_pretrained("tokenizers/gpt_neo_125m_tokenizer")
model=GptNeoTune(base_model)
l.seed_everything(41)
dialog=load_dataset('daily_dialog') # loads the  unprocessed data (open it from a file system or load it from hf datasets)
chat=preprocess_daily_dialog.return_chat_data(dialog)

wandb_logger=WandbLogger(project='lightning_neo_professional_chat',name="init_run")
train_loader=DataLoader(chat['train'],batch_size=Config.train_batch_size_lightning,shuffle=False,collate_fn=utils.MyCollator(tokenizer.get_vocab()['[pad]'],tokenizer.model_max_length))
val_loader=DataLoader(chat['validation'],batch_size=TrainConfig.eval_batch_size_lightning,shuffle=False,collate_fn=utils.MyCollator(tokenizer.get_vocab()['[pad]'],tokenizer.model_max_length))
trainer=l.Trainer(accelerator='gpu',max_epochs=15,min_epochs=3,logger=wandb_logger,strategy="ddp")
trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=val_loader)