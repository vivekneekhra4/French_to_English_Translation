from flask import Flask
from flask_restful import Api
from flask_cors import CORS
#from authenticate_voice import AuthenticateVoice
from mt_v2 import MachineTranslation


app = Flask(__name__)
CORS(app)
#cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)
app.json.sort_keys=False




@app.route('/')
def index():
    return "<h1>Flask API Server is working for Item Generation for machine translation English to Candadian French</h1>"


"""
    Routes
    ** '/api/translate'

"""


api.add_resource(MachineTranslation, '/ai/translate/fr-en')


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
