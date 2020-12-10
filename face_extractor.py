# Import libraries
import os
import cv2
import numpy as np


# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"


# Read the model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# Create directory 'faces' if it does not exist
if not os.path.exists('faces'):
	print("New directory created")
	os.makedirs('faces')

# Loop through all images and strip out faces
count = 0
for file in os.listdir(base_dir + '/images/'):
	file_name, file_extension = os.path.splitext(file)
	if (file_extension in ['.png','.jpg']):
		print("Image path: {}".format(base_dir + '/images/' + file))

		image = cv2.imread(base_dir + '/images/' + file)

		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		net.setInput(blob)
		detections = net.forward()

	for i in range(0, detections.shape[2]):
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		confidence = detections[0, 0, i, 2]

		# If confidence > 0.5, save it as a separate file
		if (confidence > 0.5):
			count += 1
			frame = image[startY:endY, startX:endX]
			cv2.imwrite(base_dir + '/faces/' + str(i) + '_' + file, frame)



	# Identify each face
		# for i in np.arange(0, detections.shape[2]):
		#
		# 	confidence = detections[0, 0, i, 2]
		#
		# 	# If confidence > 0.5, save it as a separate file
		# 	if (confidence > 0.5):
		# 		# count += 1
		# 		# # compute the (x, y)-coordinates of the bounding box for the
		# 		# # object
		# 		# box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# 		# (startX, startY, endX, endY) = box.astype("int")
		# 		#
		# 		# # draw the bounding box of the face along with the associated
		# 		# # probability
		# 		# text = "{:.2f}%".format(confidence * 100) + 'Count ' + str(count)
		# 		# y = startY - 10 if startY - 10 > 10 else startY + 10
		# 		# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		# 		# cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		# 		# # compute the (x, y)-coordinates of the bounding box for the
		# 		# # object
		# 		# box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# 		# (startX, startY, endX, endY) = box.astype("int")
		# 		#
		# 		# # draw the bouding box of the fqqqqace along with the associated
		# 		# # probability
		# 		# text = "{:.2f}%".format(confidence * 100)
		# 		# f, s, t = (cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
		# 		# (text_width, text_height) = cv2.getTextSize(text, f, s, t)[0]
		# 		# y = startY - 10 if startY - 10 > 10 else startY + 10
		# 		#
		# 		# # Extract face shot and body shot from frame\
		# 		# # creating a copy of frame
		# 		# frame_copy = frame.copy()
		# 		#
		# 		# # calculate box width
		# 		# box_width = endX - startX
		# 		#
		# 		# # calculate box height
		# 		# box_height = endY - startY
		# 		#
		# 		# # extracting face shot
		# 		# # expanding the rectangle to make a better face crop
		# 		# # expanding the top by 50%, right by 20%
		# 		# # expanding the below chin part by only 10% and left by 20%
		# 		# bottom_left, top_left, top_right, bottom_right = (startX - int(startX * (30 / 100)),
		# 		# 												  startY - int(startY * (50 / 100)),
		# 		# 												  endX + int(endX * (20 / 100)),
		# 		# 												  endY + int(endY * (10 / 100))
		# 		# 												  )
		# 		#
		# 		# face_shot = frame_copy[top_left:bottom_right, bottom_left:top_right]
		# 		# cv2.imwrite(capture_training_dir + "my-image-face_" + timestr + ".png", face_shot)
		# 		#
		# 		# # extracting face shot for ecognition
		# 		# face_shot = frame_copy[startY:endY, startX:endX]
		# 		# cv2.imwrite(capture_detection_dir + "my-image-face_" + timestr + ".png", face_shot)
		# 		#
		# 		# # Extract body from frame
		# 		#
		# 		# # expanding the rectangle to make a better face crop
		# 		# # expanding the top by 50%, right by 20%
		# 		# # expanding the below chin part by only 10% and left by 20%
		# 		# bottom_left, top_left, top_right, bottom_right = (startX - box_width,
		# 		# 												  startY - int(startY * (50 / 100)),
		# 		# 												  endX + box_width,
		# 		# 												  endY + box_height
		# 		# 												  )
		# 		#
		# 		# body_shot = frame_copy[top_left:bottom_right, bottom_left:top_right]
		# 		# cv2.imwrite(capture_training_dir + "my-image-body_" + timestr + ".png", body_shot)
		#
		# 		# idx = int(detections[0, 0, i, 1])
		# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# 		(startX, startY, endX, endY) = box.astype("int")
		# 		count += 1
		# 		frame = image[startY:endY, startX:endX]
		# 		cv2.imwrite(base_dir + "\\faces\\" + str(i) + '_' + file, frame)

print("Extracted " + str(count) + " faces from all images")