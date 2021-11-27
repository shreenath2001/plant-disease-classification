from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
import json

# Create your views here.

label = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy', 4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy', 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_', 9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot', 12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy', 15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy', 18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight', 21: 'Potato___Late_blight', 22: 'Potato___healthy', 23: 'Raspberry___healthy', 24: 'Soybean___healthy', 25: 'Squash___Powdery_mildew', 26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy', 28: 'Tomato___Bacterial_spot', 29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight', 31: 'Tomato___Leaf_Mold', 32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite', 34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 36: 'Tomato___Tomato_mosaic_virus', 37: 'Tomato___healthy'}

model = load_model('./model/model.h5', custom_objects={'KerasLayer':hub.KerasLayer})

def classify(request):
    return render(request, 'app/classify.html')

def classifyImage(request):
    if request.method == 'POST':
        image = request.FILES['image']
        fs = FileSystemStorage()
        imageName = fs.save(image.name, image)
        imageName = fs.url(imageName)
        loc = '.' + imageName
        
        img = load_img(loc, target_size=(224, 224))
        img_arry = img_to_array(img)
        img = img_arry/255.0
        ''' img_arry = img_to_array(img)
        to_pred = np.expand_dims(img_arry, axis = 0)
        prep = preprocess_input(to_pred) '''
        prep = np.asarray([img])
        prediction = model.predict(prep)[0]
        percentage = np.max(prediction)
        prediction = np.argmax(prediction)
        #print("percentage:", percentage)
        if(percentage > 0.9):
            ans = label[prediction]
        else:
            ans = "Sorry, unable to classify"

        context = {
            'imageName': imageName,
            'label': ans
        }
        return render(request, "app/classify.html", context)
