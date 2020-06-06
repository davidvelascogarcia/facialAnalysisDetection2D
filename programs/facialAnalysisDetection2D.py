
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
print("Loading facialAnalysisDetection2D module ...")


print("")
print("")
print("**************************************************************************")
print("YARP configuration:")
print("**************************************************************************")
print("")
print("")
print("Initializing YARP network ...")

# Init YARP Network
yarp.Network.init()


print("")
print("[INFO] Opening image input port with name /facialAnalysisDetection2D/img:i ...")

# Open input image port
facialAnalysisDetection2D_portIn = yarp.BufferedPortImageRgb()
facialAnalysisDetection2D_portNameIn = '/facialAnalysisDetection2D/img:i'
facialAnalysisDetection2D_portIn.open(facialAnalysisDetection2D_portNameIn)

print("")
print("[INFO] Opening image output port with name /facialAnalysisDetection2D/img:o ...")

# Open output image port
facialAnalysisDetection2D_portOut = yarp.Port()
facialAnalysisDetection2D_portNameOut = '/facialAnalysisDetection2D/img:o'
facialAnalysisDetection2D_portOut.open(facialAnalysisDetection2D_portNameOut)

print("")
print("[INFO] Opening data output port with name /facialAnalysisDetection2D/data:o ...")

# Open output data port
facialAnalysisDetection2D_portOutDet = yarp.Port()
facialAnalysisDetection2D_portNameOutDet = '/facialAnalysisDetection2D/data:o'
facialAnalysisDetection2D_portOutDet.open(facialAnalysisDetection2D_portNameOutDet)

# Create data bootle
cmd=yarp.Bottle()

# Create coordinates bootle
coordinates=yarp.Bottle()

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
print("")
print("**************************************************************************")
print("Loading models:")
print("**************************************************************************")
print("")
print("")
print("Loading models ...")
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


print("")
print("")
print("**************************************************************************")
print("Waiting for input image source:")
print("**************************************************************************")
print("")
print("")
print("Waiting input image source ...")
print("")

# Control loop
loopControlReadImage = 0

while int(loopControlReadImage) == 0:

    # Recieve image source
    frame = facialAnalysisDetection2D_portIn.read()

    print("")
    print("")
    print("**************************************************************************")
    print("Processing:")
    print("**************************************************************************")
    print("")
    print("Processing data ...")

    # Buffer processed image
    in_buf_image.copy(frame)
    assert in_buf_array.__array_interface__['data'][0] == in_buf_image.getRawImage().__int__()

    # YARP -> OpenCV
    rgb_frame = in_buf_array[:, :, ::-1]


    print("")
    print("")
    print("**************************************************************************")
    print("Analyzing image source:")
    print("**************************************************************************")
    print("")
    print("Analyzing image source ...")
    print("")

    # Analyzing frame
    facialAnalysisDetection2DResults = DeepFace.analyze(rgb_frame, models=models)
    print("")
    print("[INFO] Image source analysis done correctly.")
    print("")

    # Get time Detection
    timeDetection = datetime.datetime.now()


    # Print processed data
    print("")
    print("**************************************************************************")
    print("Results resume:")
    print("**************************************************************************")
    print("")
    print("[RESULTS] Facial analysis results:")
    print("Gender: ", facialAnalysisDetection2DResults["gender"])
    print("Age: ", facialAnalysisDetection2DResults["age"])
    print("Race: ", facialAnalysisDetection2DResults["dominant_race"])
    print("Emotion: ", facialAnalysisDetection2DResults["dominant_emotion"])
    print("[INFO] Detection time: "+ str(timeDetection))

    # Text to show in the image
    imageText = "G: " + str(facialAnalysisDetection2DResults["gender"]) + ", A: " + str(int(facialAnalysisDetection2DResults["age"])) + ", R: " + str(facialAnalysisDetection2DResults["dominant_race"]) + ", E: " + str(facialAnalysisDetection2DResults["dominant_emotion"])

    # Write processed data in the frame
    in_buf_array = cv2.rectangle(in_buf_array, (50, 50), (600, 420), (36,255,12), 1)
    cv2.putText(in_buf_array, imageText, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Sending processed detection
    cmd.clear()
    cmd.addString("Gender:")
    cmd.addString(str(facialAnalysisDetection2DResults["gender"]))
    cmd.addString("Age:")
    cmd.addString(str(facialAnalysisDetection2DResults["age"]))
    cmd.addString("Race:")
    cmd.addString(str(facialAnalysisDetection2DResults["dominant_race"]))
    cmd.addString("Emotion:")
    cmd.addString(str(facialAnalysisDetection2DResults["dominant_emotion"]))
    cmd.addString("Time:")
    cmd.addString(str(timeDetection))
    facialAnalysisDetection2D_portOutDet.write(cmd)

    # Sending processed image
    print("")
    print("[INFO] Sending processed image ...")
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
