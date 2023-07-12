FROM python:3.9.17-bullseye

WORKDIR /app

COPY requirements.txt .
COPY templates templates
COPY static static
COPY speechbrain speechbrain
COPY audios audios
COPY package-lock.json package.json tailwind.config.js ./
COPY app.py .

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]
