import os
import json
import time
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = 'deepset/deberta-v3-large-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

input_folder = '/home/azureuser/Bert/Bert_json/json_folder1'  
output_folder = '/home/azureuser/Bert/Bert_json/json_server'  

def perform_qa(paragraph, question, max_length=512, device='cuda', confidence_threshold=0.25):
    inputs = tokenizer(question, paragraph, add_special_tokens=True, return_tensors='pt', max_length=1000, padding=True, truncation=True)
    model.to(device)
    inputs.to(device)
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()
    processing_time = end_time - start_time
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    start_char = inputs[0].offsets[start_index][0]
    end_char = inputs[0].offsets[end_index][1]

    if start_char is not None and end_char is not None:
        answer = paragraph[start_char:end_char]
    else:
        answer = "Answer not found"

    start_confidence = torch.softmax(start_scores, dim=1)[0][start_index].item()
    end_confidence = torch.softmax(end_scores, dim=1)[0][end_index].item()
    answer_confidence = max(start_confidence, end_confidence)

    answer = "Answer not found" if answer_confidence < confidence_threshold else answer

    return answer, start_char, end_char, answer_confidence

def process_json_file(input_file, output_folder):
    with open(input_file, 'r') as json_file:
        input_data = json.load(json_file)

    results = []

    for item in input_data:
        paragraph = item.get('C', '')
        question = item.get('Q', '')

        answer, start_char, end_char, answer_confidence = perform_qa(paragraph, question, device='cuda')

        answer_dict = {
            'answer': answer,
            'start_index': start_char,
            'end_index': end_char,
            'score': answer_confidence
        }

        results.append(answer_dict)

    output_file = os.path.join(output_folder, os.path.basename(input_file))
    with open(output_file, 'w', encoding='utf-8') as json_output_file:
        json.dump(results, json_output_file, indent=4)

if __name__ == '__main__':
    input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.json')]

    total_start_time = time.time()
    
    for i, input_file in enumerate(input_files[:300]):
        process_json_file(input_file, output_folder)
        print(i, input_file)
        

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    print(f"Total processing time for {len(input_files)} files: {total_processing_time} seconds")

torch.cuda.empty_cache()
