from flask import Flask, render_template, redirect, url_for, request, session, flash,g
import os

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse as ap
import imutils
import cv2


app = Flask(__name__)

app.secret_key = "qwertyuiopasdfghjklzxcvbnm"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/", methods=['GET', 'POST'])
def  index():
	return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, '')
    print(target)

    if not os.path.isdir(target):
         os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        print("File name is ",filename)
        destination = "/".join([target, "test.jpg"])
        print(destination)
        file.save(destination)

#running the testNet.py
    image = cv2.imread('test.jpg')
    path = './static'
    cv2.imwrite(os.path.join(path, filename), image)
    orgCpoy = image.copy()

    image = cv2.resize(image, (28,28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    print("----------------loading model----------------")
	#model = load_model(args["model"])
    model = load_model('test_model.model')

    (oily, normal, dry) = model.predict(image)[0]

    if ((oily > normal) and (oily > dry)):
	    #print("oily skin")
	    label = "oily"
    elif ((normal > oily) and (normal > dry)):
	    #print("normal skin")
	    label = "normal"
    else:
	    #print("dry skin")
	    label = "dry"

#rendering the templates based on label

    if(label == "oily"):
        return render_template("oily.html", value = filename)

    elif(label == "normal"):
        return render_template("normal.html", value = filename)

    elif(label == "dry"):
        return render_template("dry.html", value = filename)
    else:
        return render_template("index.html", value = filename)


if __name__ == "__main__":
	app.run(debug=True)
