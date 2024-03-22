import torch
import argparse
import ast
import random
from utils import get_model

def get_questions_and_answers_from_file(input_file):
    questions = []
    answers = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i%3 == 0:
                question = ast.literal_eval(line[3:])
                questions.append(question)
            elif (i+2)%3 == 0:
                answer = ast.literal_eval(line[3:])
                answers.append(answer)

    return questions, answers

def train_soft_system_prompt(model, embeddings, tokenizer, questions, answers, epochs, lr, system_prompt_length, prompt_file_name='soft_system_prompt.pt'):
    baseline_input = '!'
    baseline_token = tokenizer(baseline_input, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0)
    baseline_embedding = embeddings[baseline_token]

    soft_system_prompt = baseline_embedding.repeat(system_prompt_length, 1).to(model.device)

    bos_token = tokenizer(tokenizer.bos_token, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0)
    bos_embeddings = embeddings[bos_token].to(model.device)

    soft_system_prompt = torch.nn.Parameter(soft_system_prompt, requires_grad=True)
    optimiser = torch.optim.Adam([soft_system_prompt], lr=lr)

    for epoch in range(epochs):
        loss = torch.zeros(1).to(model.device)
        for i, (question_list, answer_list) in enumerate(zip(questions, answers)):
            question = random.choice(question_list)
            answer = random.choice(answer_list)
            print(f'Training on question: {question} and answer: {answer}')
            input_tensor = tokenizer(question, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0)
            input_embed = embeddings[input_tensor].to(model.device)

            output_tensor = tokenizer(answer, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(0).to(model.device)
            output_embed = embeddings[output_tensor].to(model.device)

            output = model(inputs_embeds = torch.cat((bos_embeddings, soft_system_prompt, input_embed, output_embed), dim=0).unsqueeze(0))
            logits = output.logits[0,-len(output_tensor)-1:-1]
            loss += torch.nn.CrossEntropyLoss()(logits, output_tensor)

        loss.backward()
        with torch.no_grad():
            print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}')
        optimiser.step()

        if loss < 0.1:
            break

    torch.save(soft_system_prompt, prompt_file_name)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-f', type=str, help='Path to input file')
    parser.add_argument('--epochs', '-e', type=int, help='Number of epochs to train for', default=1000)
    parser.add_argument('--learning_rate', '-lr', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--soft_system_prompt_length', '-pl', type=int, help='Length of soft system prompt', default=25)
    args = parser.parse_args()
    input_file = args.input_file

    questions, answers = get_questions_and_answers_from_file(input_file)
    model, embeddings, tokenizer = get_model('llama2-7b')

    train_soft_system_prompt(model, embeddings, tokenizer, questions, answers, args.epochs, args.learning_rate, args.soft_system_prompt_length)

if __name__ == '__main__':
    main()