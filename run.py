import argparse
import csv
import cv2
import numpy as np
import torch
from torchvision.models import detection


# Define the COCO dataset classes.
CLASSES = [
	'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
	'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
	'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
	'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
	'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
	'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
	'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
	'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
	'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
	'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Generate a set of bounding box colors for each class.
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Define the model names and function calls.
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn,
	"ssd-vgg16": detection.ssd300_vgg16,
	"ssd-mobilenet": detection.ssdlite320_mobilenet_v3_large
}

# Set the device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_model(args):
	"""
	Set up the model defined by the command line arguments.
	"""

	# Load the model and set it to evaluation mode.
	model = MODELS[args["model"]](pretrained=True, progress=True,
		num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
	model.eval()

	print("Model:", args["model"])

	return model


def detect_objects(args, model):
	"""
	Detect objects on the image defined by the command line arguments
	using the given model.
	"""

	# Load the image.
	image = cv2.imread(args["image"])
	orig = image.copy()

	# Convert the image from BGR to RGB channel ordering.
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Change the image from channels last to channels first ordering.
	image = image.transpose((2, 0, 1))

	# Add the batch dimension.
	image = np.expand_dims(image, axis=0)

	# Scale the raw pixel intensities to the range [0, 1].
	image = image / 255.0

	# Convert the image to a floating point tensor.
	image = torch.FloatTensor(image)

	# Send the input to the device. 
	image = image.to(DEVICE)

	# Get the detections and predictions.
	detections = model(image)[0]

	# Loop over the detections.
	for i in range(0, len(detections["boxes"])):

		# Extract the confidence (i.e., probability) associated with the prediction.
		confidence = detections["scores"][i]

		# Filter out weak detections by ensuring a high confidence.
		if confidence > args["confidence"]:

			# Extract the index of the class label from the detections.
			idx = int(detections["labels"][i])

			# Compute the (x, y)-coordinates of the bounding box for the object.
			box = detections["boxes"][i].detach().cpu().numpy()
			(startX, startY, endX, endY) = box.astype("int")

			# Display the prediction to our terminal.
			label = "{} ({:.2f}%)".format(CLASSES[idx], confidence * 100)
			print(label)

			# Draw the bounding box and label on the image.
			cv2.rectangle(orig, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(orig, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# Show the output image.
	cv2.imshow("Output", orig)
	cv2.waitKey(0)


def read_arguments():
	"""
	Read and save the command line arguments.
	"""
	
	# Set up the argument parser.
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", type=str, required=True,
		help="Path to the input image.")
	ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
		choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet", "ssd-mobilenet"],
		help="Name of the object detection model.")
	ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
		help="Path to file containing list of categories in COCO dataset.")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="Minimum probability to filter weak detections.")
	args = vars(ap.parse_args())

	return args


def main():

	print("OBJECT DETECTION")
	print("================")
		
	# Read the arguments.
	args = read_arguments()

	# Define the model.
	model = set_model(args)

	# Perform obkect detection.
	print()
	print("OBJECTS")
	print("================")
	detect_objects(args, model)


if __name__ == "__main__":
	main()