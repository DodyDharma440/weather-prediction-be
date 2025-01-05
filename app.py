from flask import Flask 
from flask_restful import Api
from flask_cors import CORS

from controllers.home import Home
from controllers.weather import Weather


app = Flask(__name__) 
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app) 

api.add_resource(Home, '/') 
api.add_resource(Weather, '/weather') 
  
# driver function 
if __name__ == '__main__': 
    app.run(debug = True)