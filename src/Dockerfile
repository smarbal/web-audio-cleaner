FROM python:3.9.17-bullseye

WORKDIR /app

COPY requirements.txt .
COPY templates templates
COPY static static
COPY speechbrain speechbrain
COPY package-lock.json package.json tailwind.config.js ./
COPY app.py .

RUN apt update 
RUN apt install ffmpeg -y
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir

ENV FLASK_APP=app.py
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

