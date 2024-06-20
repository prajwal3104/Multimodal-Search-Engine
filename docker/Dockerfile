FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/api/app.py"]