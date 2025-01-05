from flask import jsonify
from flask_restful import Resource


class Home(Resource): 
    def get(self): 
        return jsonify({'message': 'Welcome to Weather Prediction!'}) 