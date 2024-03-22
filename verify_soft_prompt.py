import torch
import argparse
from utils import get_model

def explain_soft_system_prompt(soft_system_prompt, model, embeddings, tokenizer):

    explanation_str = ' is'
    explanation_str_tokens = tokenizer(explanation_str, return_tensors='pt').input_ids.squeeze(0)
    explanation_str_embeddings = embeddings[explanation_str_tokens]

    explanation_embeddings = torch.cat((soft_system_prompt, explanation_str_embeddings), dim=0).to(model.device)
    generated_token_ids = model.generate(inputs_embeds=explanation_embeddings.unsqueeze(0), max_new_tokens=50)
    return tokenizer.decode(generated_token_ids[0])

def main():

    parser = argparse.ArgumentParser(description='Verify Soft System Prompt')
    parser.add_argument('--file_path', type=str, help='Path to the soft system prompt file')
    parser.add_argument('--input_string', type=str, help='Input string for generating output')
    args = parser.parse_args()

    file_path = args.file_path
    input_string = args.input_string

    model, embeddings, tokenizer = get_model('llama2-7b')

    soft_system_prompt = torch.load(file_path)

    explanation_string = explain_soft_system_prompt(soft_system_prompt, model, embeddings, tokenizer)

    print(f'----------- Explanation of system prompt -------------\n\n{explanation_string}\n')

    input_str = input_string
    input_tokens = tokenizer(input_str, return_tensors='pt').input_ids
    generated_token_ids = model.generate(input_tokens.to(model.device), max_new_tokens=50)

    print(f'------------ Output without system prompt -------------\n\n{tokenizer.decode(generated_token_ids[0])}\n')

    bos_token = tokenizer(tokenizer.bos_token, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0)
    bos_embedding = embeddings[bos_token].to(model.device)
    input_embeddings = torch.cat((bos_embedding, soft_system_prompt, embeddings[input_tokens.squeeze(0)]), dim=0).unsqueeze(0).to(model.device)
    generated_token_ids = model.generate(inputs_embeds=input_embeddings, max_new_tokens=50)

    print(f'------------- Output with system prompt --------------\n\n{tokenizer.decode(generated_token_ids[0])}')

if __name__ == '__main__':
    main()