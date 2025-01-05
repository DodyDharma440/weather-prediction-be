import io
import os
from flask import  jsonify, request 
from flask_restful import Resource 
from PIL import Image

import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import img_to_array # type: ignore

model = load_model(os.path.join(os.path.dirname(__file__), '../trained_models/weather.h5'))
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