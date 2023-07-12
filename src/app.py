import re
import os
import json
import torch
import torchaudio
import uuid
import shutil
from flask_cors import CORS
from flask import Flask, request, Response, render_template, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from speechbrain.pretrained import SpectralMaskEnhancement, WaveformEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="./speechbrain/pretrained_models/metricgan-plus-voicebank",
    savedir="./speechbrain/pretrained_models/metricgan-plus-voicebank"
)

ALLOWED_EXTENSIONS = set(['mp3', 'wav', 'mp4', 'zip', 'rar'])

app = Flask(__name__)
CORS(app)
app.secret_key = 'ThisIsaSecret_CRC_60II'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'audios/request'
app.config['UPLOAD_FOLDER_RESULT'] = 'audios/clean'

    


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/audio", methods=["POST"])
def analyze(): 
    files = request.files.getlist('files[]')
    if files == [] :
        flash("No files uploaded")  
        return 'Error' 
    # TODO: Add support for archives (.zip)  
    random_uuid  = uuid.uuid4() 
    request_path = os.path.join(app.config['UPLOAD_FOLDER'], str(random_uuid))
    os.makedirs(request_path, mode=0o777, exist_ok=True)
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            audio_path = os.path.join(request_path, file.filename)
            file.save(audio_path) 
    clean_folder(request_path)
    return 'Files uploaded successfully!' 
         

@app.route("/history", methods=["GET"])
def history(): 
    with open('static/results.json', 'r') as f:
        data = json.load(f) 
    return render_template('history.html', history=data)

@app.route("/result/<uuid>", methods=["GET"])
def result(uuid): 
    pdf = is_pdf(uuid)
    page = request.args.get('page', default=0, type=(int)) 
    with open('static/results.json', 'r') as f:
        data = json.load(f)
    result = data[uuid]['result']
    result_image = data[uuid]['result_img']
    return render_template('result.html', result=result, result_image=result_image, pdf=pdf, page=page,document=document)

    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_audio(audio):
    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(audio, lengths=torch.tensor([1.]))
    return enhanced

def clean_folder(folder_path):
    clean_dir_path = os.path.join(app.config['UPLOAD_FOLDER_RESULT'], folder_path.split('/')[-1])
    os.makedirs(clean_dir_path, exist_ok=True)
    os.chmod(clean_dir_path, 0o777)
    for filename in os.listdir(str(folder_path)):
        audio_path = os.path.join(str(folder_path), str(filename))
        noisy = enhance_model.load_audio(audio_path).unsqueeze(0)
        enhanced = clean_audio(noisy)
        clean_filename = os.path.splitext(filename)[0] +'.wav' 
        clean_filepath = os.path.join(clean_dir_path, clean_filename)   
        torchaudio.save(clean_filepath, enhanced.cpu(), 16000)
        os.remove(filename) # When processing the file, the model/framework seems to create a copy in the root directory, so I delete it after processing

        

if __name__ == '__main__':
    app.debug = True   # Change this when in production 
    app.run()
    
    

