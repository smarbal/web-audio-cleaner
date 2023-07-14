import re
import os
import json
import torch
import torchaudio
import uuid
import shutil
import zipfile
from flask_cors import CORS
from flask import Flask, request, Response, render_template, flash, redirect, url_for, session, get_flashed_messages, abort
from werkzeug.utils import secure_filename
from speechbrain.pretrained import SpectralMaskEnhancement, WaveformEnhancement
from pydub import AudioSegment
from datetime import datetime

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="./speechbrain/pretrained_models/metricgan-plus-voicebank",
    savedir="./speechbrain/pretrained_models/metricgan-plus-voicebank"
)

ALLOWED_EXTENSIONS = set(['mp3', 'wav', 'mp4', 'aac', 'zip'])

app = Flask(__name__)
CORS(app)
app.secret_key = 'ThisIsaSecret_CRC_60II'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'static/audios'


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/audio", methods=["POST"])
def analyze(): 
    files = request.files.getlist('files[]')
    if files == [] :
        flash("No files uploaded")
        abort(400)
    random_uuid  = str(uuid.uuid4())[:11] 
    #Create necessary folders
    request_path = os.path.join(app.config['UPLOAD_FOLDER'], random_uuid)
    original_path = os.path.join(request_path, 'original')
    os.makedirs(original_path, mode=0o777, exist_ok=True)
    clean_dir_path = os.path.join(request_path, 'clean')
    os.makedirs(clean_dir_path, mode=0o777, exist_ok=True)
    # Save files in request folder
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            audio_path = os.path.join(original_path, file.filename)
            file.save(audio_path) 
        else: 
            flash("Wrong file type.", 'error')
            abort(400)
    for filename in os.listdir(original_path):  # Zip check could be done simpler and earlier but the check messed up the audio files. 
        file_path = os.path.join(original_path, filename)
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(original_path)
            os.remove(file_path)
    # Process audio
    try:
        clean_folder(original_path, clean_dir_path)
    except: 
        flash("Error processing the audio. Please retry.")  
        abort(400)
    # Format the cleaned data 
    shutil.make_archive(os.path.join(request_path, random_uuid), 'zip', clean_dir_path)
    combined = AudioSegment.empty()
    for filename in os.listdir(clean_dir_path):
        audio_path = os.path.join(clean_dir_path, filename)
        sound = AudioSegment.from_file(audio_path, format="wav")
        combined += sound
    combined.export(os.path.join(request_path, 'FULL_' + random_uuid + '.wav'), format="wav")
    save_metadata(clean_dir_path, random_uuid, len(os.listdir(clean_dir_path)), combined.duration_seconds)
    # Flash info about being done 
    return redirect(url_for('result', uuid=random_uuid)) 

@app.route("/history", methods=["GET"])
def history(): 
    with open('static/results.json', 'r') as f:
        data = json.load(f) 
    items = list(data.keys())[::-1] # To have the latest request first in the list
    return render_template('history.html', history=data, items=items) 

@app.route("/result/<uuid>", methods=["GET"])
def result(uuid):
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], uuid, uuid + '.zip')
    full_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], uuid, 'FULL_' + uuid + '.wav')
    return render_template('result.html', uuid=uuid, zip_path=zip_path, full_audio_path=full_audio_path)

@app.route("/delete", methods=["GET"])
def delete():
    for folder in os.listdir(app.config['UPLOAD_FOLDER']):
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path) 
    with open("static/results.json", "w") as write_file:
        json.dump({}, write_file) 
    return redirect(url_for('index')) 
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_audio(audio):
    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(audio, lengths=torch.tensor([1.]))
    return enhanced

def clean_folder(folder_path, clean_dir_path):
    for filename in os.listdir(str(folder_path)):
        audio_path = os.path.join(str(folder_path), str(filename))
        noisy = enhance_model.load_audio(audio_path).unsqueeze(0)
        enhanced = clean_audio(noisy)
        clean_filename = os.path.splitext(filename)[0] +'.wav' 
        clean_filepath = os.path.join(clean_dir_path, clean_filename)   
        torchaudio.save(clean_filepath, enhanced.cpu(), 16000)
        os.remove(filename) # When processing the file, the model/framework seems to create a copy in the root directory, so I delete it after processing

def parse_filename(filename):
    parts = filename.split("_")
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    hour = int(parts[3])
    minute = int(parts[4])
    second = int(parts[5])
    return datetime(year, month, day, hour, minute, second)

def find_first_and_last_dates(directory):
    filenames = os.listdir(directory)
    if not filenames:
        return None, None
    
    dates = [parse_filename(filename) for filename in filenames]
    dates.sort()
    first_date = dates[0].strftime("%Y-%m-%d %H:%M:%S")
    last_date = dates[-1].strftime("%Y-%m-%d %H:%M:%S")
    return first_date, last_date

def save_metadata(clean_path, uuid, n_files, duration):
    first_date, last_date = find_first_and_last_dates(clean_path)
    new_data = { 'n_files' : n_files, 
                       'duration' : duration,
                       'first_date': first_date,
                       'last_date': last_date
                }
    # Open the already existing data and update it with the new one             
    with open("static/results.json", "r") as read_file:
        data = json.load(read_file)
        data[uuid] = new_data

    with open("static/results.json", "w") as write_file:
        json.dump(data, write_file) 

if __name__ == '__main__':
    app.debug = False   # Change this when in production 
    app.run()
    
    

