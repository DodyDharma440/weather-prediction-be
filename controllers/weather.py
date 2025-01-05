import io
import os
from flask import  jsonify, request 
from flask_restful import Resource 
from PIL import Image

import numpy as np
import requests
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import img_to_array # type: ignore

from tensorflow.python.lib.io import file_io

def download_model(url, save_path):
    # Cek apakah model sudah ada
    if not os.path.exists(save_path):
        print("Downloading model...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error jika gagal
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully!")
    else:
        print("Model already exists.")

# URL model (ubah ke link model Anda)
model_url = "https://drive.google.com/file/d/1GiB7gqNFAeKIQetvGI0h9Gby5q64TaUa/view?usp=sharing"
model_path = "trained_models/weather.h5"

# Unduh model jika belum ada
download_model(model_url, model_path)


# model = load_model(os.path.join(os.path.dirname(__file__), '../trained_models/weather.h5'))
model = load_model(os.path.join(os.path.dirname(__file__), "../% s" % model_path))

class_names = ['cloudy', 'rain', 'shine', 'sunrise']

class Weather(Resource):
    def post(self):
        try:
        # Pastikan file dikirim dalam body request
            if 'image' not in request.files:
                return jsonify({"error": "No image file found in the request."}), 400

            # Ambil file gambar
            image_file = request.files['image']
            image_stream = io.BytesIO(image_file.read())
            img = Image.open(image_stream).convert('RGB')
                
            # Load dan preprocess gambar langsung dari file
            img = img.resize((128, 128)) 
            img = img_to_array(img)
            img = img.reshape(1, 128, 128, 3)

            # Lakukan prediksi
            predictions = model.predict(img)

            # Ambil prediksi terbaik
            pred = predictions[0]
            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred)
            
            return jsonify({
                "predicted_class": predicted_class,
                "confidence": float(confidence)
            })

        except Exception as e:
            print('err => ', e)
            return jsonify({"error": 'Something wrong'}), 500