from transformers import LlamaTokenizer,AutoTokenizer
from utils import add_special_tokens


open_llama_3b_checkpoint="openlm-research/open_llama_3b"
gpt_neo_125m_checkpoint="EleutherAI/gpt-neo-125m"

llama_tokenizer=add_special_tokens(LlamaTokenizer.from_pretrained(open_llama_3b_checkpoint))
gpt_neo_tokenizer=add_special_tokens(AutoTokenizer.from_pretrained(gpt_neo_125m_checkpoint))


if __name__=="__main__":
    llama_tokenizer.save_pretrained("tokenizers/llama_3b_tokenizer")
    gpt_neo_tokenizer.save_pretrained("tokenizers/gpt_neo_125m_tokenizer")
    