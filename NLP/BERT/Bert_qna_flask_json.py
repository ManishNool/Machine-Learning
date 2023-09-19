import os
import json
import time
import torch
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='/home/azureuser/Bert/Bert_json/templates')

model_name = 'deepset/deberta-v3-large-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

UPLOAD_FOLDER = '/home/azureuser/Bert/Bert_json/uploads'
ALLOWED_EXTENSIONS = {'json'}  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            with open(file_path, 'r') as json_file:
                input_data = json.load(json_file)

            results = []


            for item in input_data:
                paragraph = item.get('t', '')
                question = item.get('q', '')

                answer_gpu,start_char, end_char, answer_confidence = perform_qa(paragraph, question, device='cuda')

                answer_dict = {
                    'answer': answer_gpu,
                    'start_index': start_char,
                    'end_index': end_char,
                    'score': answer_confidence
                }

                results.append(answer_dict)

            output_data = json.dumps(results, indent=4)

            with open(file_path, 'w', encoding = 'utf-8') as json_output_file:
                json_output_file.write(output_data)

            return render_template(
                "index.html",
                results=results,
            )

    return render_template("index.html")

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5005')
