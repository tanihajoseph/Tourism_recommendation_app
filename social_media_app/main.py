import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
import os 

app = Flask(__name__)
bootstrap = Bootstrap(app)

IMG_FOLDER = os.path.join('static', 'display_images')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/', methods=['GET','POST'])
def save():

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('new.html',func=save)
    
if __name__ == '__main__':
    app.run(debug=True)