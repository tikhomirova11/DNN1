# Import libraries
import os
import cv2
import numpy as np

# Define paths
base_dir = os.path.dirname(__file__)
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"


# Read the model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# Create directory 'updated_images' if it does not exist
if not os.path.exists('updated_images'):
	print("New directory created")
	os.makedirs('updated_images')

# Loop through all images and save images with marked faces
for file in os.listdir(base_dir + '/images'):
	file_name, file_extension = os.path.splitext(file)
	if (file_extension in ['.png','.jpg']):
		print("Image path: {}".format(base_dir + '/images/' + file))

		image = cv2.imread(base_dir + '/images/' + file)

		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		net.setInput(blob)
		detections = net.forward()

		# Create frame around face
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if (confidence > 0.5):
				# compute the (x, y)-coordinates of the bounding box for the
				# object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the bounding box of the face along with the associated
				# probability
				text = "{:.2f}%".format(confidence * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(image, (startX, startY), (endX, endY),
							  (0, 0, 255), 2)
				cv2.putText(image, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.imwrite(base_dir + '/updated_images/' + file, image)
		print("Image " + file + " converted successfully")

		# for i in range(0, detections.shape[2]):
		# 	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# 	(startX, startY, endX, endY) = box.astype("int")
		#
		# 	confidence = detections[0, 0, i, 2]
		#
		# 	# If confidence > 0.5, save it as a separate file
		# 	if (confidence > 0.5):
		# 		count += 1
		# 		frame = image[startY:endY, startX:endX]
		# 		cv2.imwrite(base_dir + '/faces/' + '_' + file, frame)
		# print("Extracted " + str(count) + " faces from all images")