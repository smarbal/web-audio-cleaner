import re
import os
import json
import torch
import torchaudio
from flask_cors import CORS
from flask import Flask, request, Response, render_template, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from speechbrain.pretrained import SpectralMaskEnhancement, WaveformEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="./speechbrain/pretrained_models/metricgan-plus-voicebank",
    savedir="./speechbrain/pretrained_models/metricgan-plus-voicebank"
)
UPLOAD_FOLDER = './static/audio'  
UPLOAD_FOLDER_RESULT = './static/clean_audio'
ALLOWED_EXTENSIONS = set(['mp3', 'wav', 'mp4', 'zip', 'rar'])

app = Flask(__name__)
CORS(app)
app.secret_key = 'ThisIsaSecret_CRC_60II'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_RESULT'] = UPLOAD_FOLDER_RESULT

    


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/audio", methods=["POST"])
def analyze(): 
    files = request.files.getlist('files[]')

    # if 'file' not in request.files: 
    #     flash('No file part')
    #     print("No files") 
        # return redirect(url_for('index'))
    print(files)
    for file in files:
        # Process each file as needed
        file.save(f'audios/{file.filename[0:28]}')

    return 'Files uploaded successfully!' 

    # if user does not select file, browser also
    # submit a empty part without filename
    # if file.filename == '':
    #     flash('No selected file')
    #     # return redirect(request.url)
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(img_path)
    #     pages = 0
        
        # if filename.rsplit('.', 1)[1].lower() == 'pdf' : 
        #     result, pages = analyze_paddle_pdf(img_path, lang)
        #     format_result_pdf(result)

        # else :
        #     result = analyze_paddle(img_path, lang)
        #     draw_box(filename, result)
        #     format_result(result)

        # if request.form['spellcheck'] == 'true' :  
        #     spellcheck(result, lang) 
 
        # save_json(filename, result, pages)

        # return redirect(url_for('result', document=filename)) 
    return 'Error'


@app.route("/history", methods=["GET"])
def history(): 
    with open('static/results.json', 'r') as f:
        data = json.load(f) 
    return render_template('history.html', history=data)

@app.route("/result/<document>", methods=["GET"])
def result(document): 
    pdf = is_pdf(document)
    page = request.args.get('page', default=0, type=(int)) 
    with open('static/results.json', 'r') as f:
        data = json.load(f)
    result = data[document]['result']
    result_image = data[document]['result_img']
    return render_template('result.html', result=result, result_image=result_image, pdf=pdf, page=page,document=document)

    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.debug = True   # Change this when in production
    app.run()
    
    

