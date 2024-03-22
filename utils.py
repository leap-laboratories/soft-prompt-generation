import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(name='llama2-7b'):

    if name == 'llama2-7b':
        model_path = 'meta-llama/Llama-2-7b-chat-hf'
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left')
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
        embeddings = model.model.embed_tokens.weight.detach()
    else:
        raise "Not yet implemented. Please load model, tokenizer and embedding table manually"