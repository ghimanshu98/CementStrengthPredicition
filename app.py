from crypt import methods
from flask import Flask, request, render_template, Response
import os
from flask_cors import CORS, cross_origin

# creating Flask app
app = Flask(__name__)


# For homepage
@app.route('/', methods = ['POST', 'GET'])
@cross_origin()
def homepage():
    pass


# For Training
@app.route('/train', methods=['POST'])
@cross_origin()
def train():
    pass


# For Prediction
@app.route('/predict', methods = ['POST'])
@cross_origin()
def predict():
    pass


if __name__ == '__main__':
    app.run()