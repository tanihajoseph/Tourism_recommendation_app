from flask import Flask, render_template, request
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from base64 import b64encode
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import random

app = Flask(__name__)

#global sess
#sess = tf.Session()
#global graph
#graph = tf.get_default_graph()
bootstrap = Bootstrap(app)
model = tf.keras.models.load_model("models/IMAC_saved_model_using_conv-4-layers.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'C:/Users/app/static/liked_images/',
        target_size=(150, 150),
        batch_size=1,
        class_mode=None, 
        shuffle=False)
        
pred = model.predict_generator(test_generator)
print(pred)
y_pred = np.array([])
for i in range(len(pred)): 
  if pred[i]>0.5:
    #print(i)
    y_pred=np.append(y_pred,1)
    #print(" is sad")
  else:
    #print(i)
    y_pred=np.append(y_pred,0)
    #print(" is happy")
print(y_pred)   
num_zeros=0
num_ones=0
total=len(y_pred)
for m in y_pred:
  if m==0:
    num_zeros=num_zeros+1
  elif m==1:
    num_ones=num_ones+1

percent_0=(num_zeros/total)*100
percent_1=100-percent_0
mood=0
if percent_0>=70:
  mood=1
  print("SUGGEST HAPPY")
elif percent_1>=70:
  mood=2
  print("SUGGEST SAD")
else:
  mood=3
  print("SUGGEST NEUTRAL")

#@app.route('/', methods=['GET','POST'])
#def predict():

with open("C:/Users/tourism/south_india_tourism.json", 'r', encoding="utf8") as f:
    datastore = json.load(f)  

sno=[]
place=[]
img_url=[]
description=[]

if mood==1:
  for k in datastore["happy"]:
    for i in k:
      if(i=="sno"):
        sno.append(k[i])
      if(i=="place"):
        place.append(k[i])
      if(i=="description"):
        description.append(k[i])

elif mood==2:
  for k in datastore["sad"]:
    for i in k:
      if(i=="sno"):
        sno.append(k[i])
      if(i=="place"):
        place.append(k[i])
      if(i=="description"):
        description.append(k[i])

else:
  for k in datastore["neutral"]:
    for i in k:
      if(i=="sno"):
        sno.append(k[i])
      if(i=="place"):
        place.append(k[i])
      if(i=="description"):
        description.append(k[i])

		
rand=random.choice(sno)
img_file=""
if mood==1:
	img_file=img_file+"happy_"+str(rand)+".jpg"
elif mood==2:
	img_file=img_file+"sad_"+str(rand)+".jpg"
else:
	img_file=img_file+"neutral_"+str(rand)+".jpg"

print(img_file)	
place1=place[rand-1]
print(place[rand-1])

import numpy as np
import urllib
import cv2
#from google.colab.patches import cv2_imshow
#url=img_url[rand-1]
#print(url)
#resp = urllib.request.urlopen(url)
#image = np.asarray(bytearray(resp.read()), dtype="uint8")
#image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#cv2.imshow('image', image) #use in jupyter
#cv2_imshow(image) #only for colab

desc=description[rand-1]
print(desc)

IMG_FOLDER = os.path.join('static', 'images_places')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

	
@app.route('/', methods=['GET','POST'])
def initialise():
    return render_template('index.html')
    

@app.route('/predict', methods=['GET','POST'])
def predict():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], img_file)
    return render_template('result.html',place=place1, description=desc, img_path=full_filename)
    
if __name__ == '__main__':
    app.run(debug=True)