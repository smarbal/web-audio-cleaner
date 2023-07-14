# Web-audio-cleaner
Web application to denoise and enhance speech in audio data with ML models. 
The application uses Flask as web server and Speechbrain with the MetricGAN+ model for the audio processing. 
Tailwind and Flowbite are used for the presentation (CSS).

## Installation 
The application is served in a Docker container. Simply run: 
`sudo docker-compose up`

Alternatively, you can run the following commands: 
``` 
pip install -r requirements.txt --no-cache-dir
sudo apt update
sudo apt install ffmpeg
cd src 
flask run --host=0.0.0.0 --port=3000
```

## Developement 

Follow the same steps as the installation and run `npm install` in the src directory. 
Note that you can activate the debug flags in the `app.py` and the `docker-compose.yml` files to automatically take your changes into account. 

If you decide to change the HTML and CSS, note that the project uses [Tailwind](https://tailwindcss.com/).
You will thus need to recompile the `output.css` file or you will not see any change in style. 

`npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css --watch`


## Audio enhacement 

The application should support multiple file formats and `.zip` archives too. 
The goal is to be able to process an ambient recording in order to reduce the noise and better understand the speaking voices. 
In the current state of the art, machine learning algortihms seem to perform the best at this task. 
Multiple existing options were explored: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), [SpeechBrain](https://github.com/speechbrain/speechbrain), [VoiceFixer](https://github.com/haoheliu/voicefixer), [noisereduce](https://pypi.org/project/noisereduce/). 
From our testing, on our specific samples which the application was built for, [SpeechBrain](https://github.com/speechbrain/speechbrain) obtained the best results, reducing noise while keeping some ambient context and it had relatively low processing times. 

The results can be exported in zip archive, or in a single WAV file where all the audios have been joined together which fitted our need since we have a lot of small successive audios.

The audios must ideally be formated like this : `year_month_day_hour_minute_second.wav`. 
The names are used to see the start and end times of the processed files in the history view. Also when the files are processed, `listdir` automatically orders them by ASCII code, so that buts them in the right order if the names are correctly formatted. 
If the full audio file isn't right, the zip archive will not suffer the consequences of bad parsing of the dates.  


