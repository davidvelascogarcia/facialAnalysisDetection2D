
'''
  * ************************************************************
  *      Program: Facial Analysis Detection 2D Module
  *      Type: Python
  *      Author: David Velasco Garcia @davidvelascogarcia
  * ************************************************************
  *
  * | INPUT PORT                           | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /facialAnalysisDetection2D/img:i     | Input image                                             |
  *
  *
  * | OUTPUT PORT                          | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /facialAnalysisDetection2D/img:o     | Output image with facial detection analysis             |
  * | /facialAnalysisDetection2D/data:o    | Output result, facial analysis data                     |
  *
'''
# Libraries
import cv2
import datetime
from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
import json
import numpy as np
import time
import yarp

print("**************************************************************************")
print("**************************************************************************")
print("                   Program: Facial Analysis Detector 2D                   ")
print("                     Author: David Velasco Garcia                         ")
print("                             @davidvelascogarcia                          ")
print("**************************************************************************")
print("**************************************************************************")

print("")
print("Starting system ...")
print("")

print("")
print("Loading facialAnalysisDetection2D module ...")
print("")

print("")
print("**************************************************************************")
print("YARP configuration:")
print("**************************************************************************")
print("")
print("Initializing YARP network ...")
print("")

# Init YARP Network
yarp.Network.init()

print("")
print("[INFO] Opening image input port with name /facialAnalysisDetection2D/img:i ...")
print("")

# Open facialAnalysisDetection2D input image port
facialAnalysisDetection2D_portIn = yarp.BufferedPortImageRgb()
facialAnalysisDetection2D_portNameIn = '/facialAnalysisDetection2D/img:i'
facialAnalysisDetection2D_portIn.open(facialAnalysisDetection2D_portNameIn)

print("")
print("[INFO] Opening image output port with name /facialAnalysisDetection2D/img:o ...")
print("")

# Open facialAnalysisDetection2D output image port
facialAnalysisDetection2D_portOut = yarp.Port()
facialAnalysisDetection2D_portNameOut = '/facialAnalysisDetection2D/img:o'
facialAnalysisDetection2D_portOut.open(facialAnalysisDetection2D_portNameOut)

print("")
print("[INFO] Opening data output port with name /facialAnalysisDetection2D/data:o ...")
print("")

# Open facialAnalysisDetection2D output data port
facialAnalysisDetection2D_portOutDet = yarp.Port()
facialAnalysisDetection2D_portNameOutDet = '/facialAnalysisDetection2D/data:o'
facialAnalysisDetection2D_portOutDet.open(facialAnalysisDetection2D_portNameOutDet)

# Create data bootle
outputBottleFacialAnalysisDetection2D = yarp.Bottle()

# Image size
image_w = 640
image_h = 480

# Prepare input image buffer
in_buf_array = np.ones((image_h, image_w, 3), np.uint8)
in_buf_image = yarp.ImageRgb()
in_buf_image.resize(image_w, image_h)
in_buf_image.setExternal(in_buf_array.data, in_buf_array.shape[1], in_buf_array.shape[0])

# Prepare output image buffer
out_buf_image = yarp.ImageRgb()
out_buf_image.resize(image_w, image_h)
out_buf_array = np.zeros((image_h, image_w, 3), np.uint8)
out_buf_image.setExternal(out_buf_array.data, out_buf_array.shape[1], out_buf_array.shape[0])

print("")
print("[INFO] YARP network configured correctly.")
print("")

print("")
print("**************************************************************************")
print("Loading models:")
print("**************************************************************************")
print("")
print("[INFO] Loading models at " + str(datetime.datetime.now()) + " ...")
print("")

# Load models
models = {}
models["emotion"] = Emotion.loadModel()
models["age"] = Age.loadModel()
models["gender"] = Gender.loadModel()
models["race"] = Race.loadModel()

print("")
print("[INFO] Models loaded correctly.")
print("")

# Control loop
loopControlReadImage = 0

while int(loopControlReadImage) == 0:

    print("")
    print("**************************************************************************")
    print("Waiting for input image source:")
    print("**************************************************************************")
    print("")
    print("[INFO] Waiting input image source at " + str(datetime.datetime.now()) + " ...")
    print("")

    # Recieve image source
    frame = facialAnalysisDetection2D_portIn.read()

    print("")
    print("**************************************************************************")
    print("Processing input image data:")
    print("**************************************************************************")
    print("")
    print("[INFO] Processing input image data at " + str(datetime.datetime.now()) + " ...")
    print("")

    # Buffer processed image
    in_buf_image.copy(frame)
    assert in_buf_array.__array_interface__['data'][0] == in_buf_image.getRawImage().__int__()

    # YARP -> OpenCV
    rgb_frame = in_buf_array[:, :, ::-1]

    print("")
    print("**************************************************************************")
    print("Analyzing image source:")
    print("**************************************************************************")
    print("")
    print("[INFO] Analyzing image source at " + str(datetime.datetime.now()) + " ...")
    print("")

    # Analyzing frame
    facialAnalysisDetection2DResults = DeepFace.analyze(rgb_frame, models=models)
    print(facialAnalysisDetection2DResults)

    print("")
    print("[INFO] Image source analysis done correctly.")
    print("")

    # Extracted analysis detection
    genderDetection = facialAnalysisDetection2DResults["gender"]
    ageDetection = facialAnalysisDetection2DResults["age"]
    dominantRaceDetection = facialAnalysisDetection2DResults["dominant_race"]
    dominantEmotionDetection = facialAnalysisDetection2DResults["dominant_emotion"]


    # Print processed data
    print("")
    print("**************************************************************************")
    print("Resume results:")
    print("**************************************************************************")
    print("")
    print("[RESULTS] Facial analysis results:")
    print("")
    print("[GENDER] Gender: " + str(genderDetection))
    print("[AGE] Age: " + str(ageDetection))
    print("[RACE] Race: " + str(dominantRaceDetection))
    print("[EMOTION] Emotion: " + str(dominantEmotionDetection))
    print("[DATE] Detection time: " + str(datetime.datetime.now()))
    print("")

    # Text to show in the image
    imageText = "G: " + str(genderDetection) + ", A: " + str(int(ageDetection)) + ", R: " + str(dominantRaceDetection) + ", E: " + str(dominantEmotionDetection)

    # Write processed data in the frame
    in_buf_array = cv2.rectangle(in_buf_array, (50, 50), (600, 420), (36,255,12), 1)
    cv2.putText(in_buf_array, imageText, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Sending processed detection
    outputBottleFacialAnalysisDetection2D.clear()
    outputBottleFacialAnalysisDetection2D.addString("GENDER:")
    outputBottleFacialAnalysisDetection2D.addString(str(facialAnalysisDetection2DResults["gender"]))
    outputBottleFacialAnalysisDetection2D.addString("AGE:")
    outputBottleFacialAnalysisDetection2D.addString(str(facialAnalysisDetection2DResults["age"]))
    outputBottleFacialAnalysisDetection2D.addString("RACE:")
    outputBottleFacialAnalysisDetection2D.addString(str(facialAnalysisDetection2DResults["dominant_race"]))
    outputBottleFacialAnalysisDetection2D.addString("EMOTION:")
    outputBottleFacialAnalysisDetection2D.addString(str(facialAnalysisDetection2DResults["dominant_emotion"]))
    outputBottleFacialAnalysisDetection2D.addString("DATE:")
    outputBottleFacialAnalysisDetection2D.addString(str(datetime.datetime.now()))
    facialAnalysisDetection2D_portOutDet.write(outputBottleFacialAnalysisDetection2D)

    # Sending processed image
    print("")
    print("[INFO] Sending processed image at " + str(datetime.datetime.now()) + " ...")
    print("")
    out_buf_array[:,:] = in_buf_array
    facialAnalysisDetection2D_portOut.write(out_buf_image)

# Close ports
print("[INFO] Closing ports ...")
facialAnalysisDetection2D_portIn.close()
facialAnalysisDetection2D_portOut.close()
facialAnalysisDetection2D_portOutDet.close()

print("")
print("")
print("**************************************************************************")
print("Program finished")
print("**************************************************************************")
print("")
print("facialAnalysisDetection2D program closed correctly.")
print("")
