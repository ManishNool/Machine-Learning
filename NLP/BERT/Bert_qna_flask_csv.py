import os
import time
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import torch

app = Flask(__name__, template_folder='/home/azureuser/Bert/Bert_index/templates') 

model_name = 'deepset/deberta-v3-large-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

UPLOAD_FOLDER = '/home/azureuser/Bert/Bert_index/uploads' 
ALLOWED_EXTENSIONS = {'csv'}
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

            df = pd.read_csv(file_path)
            results = []
            total_time_gpu = 0  

            for index, row in df.iterrows():
                paragraph = row['paragraph']
                question = row['question']
    

                answer_gpu, time_taken_gpu, start_index, end_index, answer_confidence = perform_qa(paragraph, question, device='cuda')

                results.append([answer_gpu, time_taken_gpu, start_index, end_index, answer_confidence])

                total_time_gpu += time_taken_gpu

            new_columns = ['answer', 'time_taken_gpu', 'start_index', 'end_index', 'answer_confidence']
            
            result_df = pd.DataFrame(results, columns=new_columns)
            
            df = pd.concat([df, result_df], axis=1)
            
            df.drop(columns=['answer', 'time_taken_gpu', 'start_index', 'end_index', 'answer_confidence'], inplace=True)
            
            df['answer'] = [result[0] for result in results]
            df['start_index'] = [result[2] for result in results]
            df['end_index'] = [result[3] for result in results]
            df['answer_confidence'] = [result[4] for result in results]
            df['time_taken_gpu'] = [result[1] for result in results]
            
            df.to_csv(file_path, index=False)

            return render_template(
                "index.html",
                results=results,
                total_time_gpu=total_time_gpu
            )
    return render_template("index.html")

def perform_qa(paragraph, question, max_length=512, device='cuda', confidence_threshold=0.25):
    inputs = tokenizer(question, paragraph, add_special_tokens=True, return_tensors='pt', max_length=1000, truncation=True)
    model.to(device)
    inputs.to(device)
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()
    processing_time = end_time - start_time
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    start_confidence = torch.softmax(start_scores, dim=1)[0][start_index].item()
    end_confidence = torch.softmax(end_scores, dim=1)[0][end_index].item()
    answer_confidence = max(start_confidence, end_confidence)

    answer = "Answer not found" if answer_confidence < confidence_threshold else tokenizer.decode(inputs['input_ids'][0][start_index:end_index+1])
    
    return answer, processing_time, start_index.item(), end_index.item(), answer_confidence

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
