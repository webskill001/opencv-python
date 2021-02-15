#
# @filename object_detection.py
#
# @author sumit chouhan
#
# @desc this code is used to detect objects from image 
#
# object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import necessary pakages
import numpy as np
import argparse
import cv2

num_of_person_count = 0			# for counting number of persons detected

# contruct argument parser for user input from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detected")
args = vars(ap.parse_args())

# initailize list of class to be detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", 
			"boat", "bottle", "bus", "car", "cat", "chair",
			"cow", "diningtable", "dog", "horse", "motorbike",
			"person", "pottedplant", "sheep", "sofa", "train",
			"tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# read the frame from user input
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (600, 600), 127.5)

# pass the blob through the network and obtain the detected and
# predictions
print("[INFO] performing object detection...")
net.setInput(blob)
detected = net.forward()

# loop over the detected for object detection
for i in np.arange(0, detected.shape[2]):
	# extract the confidence associated with the prediction
	confidence = detected[0, 0, i, 2]

	# filter out weak confidence 
	if confidence > args["confidence"]:
		# detect class level from detected and draw bounding boxes around it
		idx = int(detected[0, 0, i, 1])

		draw = detected[0, 0, i, 3:7] * np.array([w, h, w, h])

		(startX, startY, endX, endY) = draw.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))

		# sort persons from detected list
		# 15 : persons 
		if idx == 15:
			num_of_person_count += 1

		cv2.rectangle(image, (startX, startY), (endX, endY),
			(255,0,0), 1)
			
		if startY - 15 > 15:
			y = startY - 15
		else:
			startY + 15

		cv2.putText(image, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


cv2.putText(image, "Number of persons = " + str(num_of_person_count), (20,25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)