# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse as ap
import imutils
import cv2

#argParser = ap.ArgumentParser()
#argParser.add_argument("-m", "--model", required = True)
#argParser.add_argument("-i", "--image", required = True)
#args = vars(argParser.parse_args())

#image = cv2.imread(args["image"])
image = cv2.imread('test.jpg')
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
    print("oily skin")
    label = "oily"
elif ((normal > oily) and (normal > dry)):
    print("normal skin")
    label = "normal"
else:
    print("dry skin")
    label = "dry"



"""
Dry Skin: Olay Total Effects, Olay Regenerist, Olay White Radiance Brightening Intensive Cream, Olay Natural White 7 IN ONE
Oily Skin: Olay Clarity Fresh Cleanser, Olay Total Effects Cream + Serum Duo SPF-15, Olay Gentle Foaming Face Wash, Olay Moisture Balance Foaming Face Wash
Normaal Skin: Olay Natural White Night Cream, Olay Age Protect Anti-Ageing Cream,  Olay Total Effects Normal Day Cream SPF 15, Olay Total Effect 7 in 1 Day Cream Touch Of Foundation SPF 15"""
