########################################################################
# Grocery Recognition
# CREATED BY: NEEL PATEL
# DATE CREATED: NOV 2019
########################################################################

# Usage example:  python main.py --list=grocery.txt --video=test.mp4     <--VIDEO
#                 python main.py --list=grocery.txt --image=test.jpg     <--IMAGE
#                 python main.py --list=grocery.txt                      <--WEBCAM


# Import computer vision package
import cv2
import numpy as np
# Import standard packages
import argparse
import sys
import os.path
from collections import Counter


def getGroceryList(filename):
    groceryList = []
    g_list = getListfromFile(filename)

    for item in g_list:
        product, quantity = item.split('|')
        groceryList.append([product, quantity])

    return groceryList

def getRemainingGrocery(groceryList, detectedProducts):
    result = groceryList[:]
    for d_item in detectedProducts:
        for item in result:
            # if label match
            if item[0] == d_item[0]:
                qty = int(item[1]) - d_item[1]
                if qty == 0:
                    result.remove(item)
                else:
                    item[1] = str(qty) # quanity

    return result

def printGroceryList(groceryList, title='Grocery List'):
    print('--------------------------------------')
    print(title)
    print('--------------------------------------')
    for item in groceryList:
        print(item)
    print('\n')

def getListfromFile(filename):
    with open(filename) as f:
        _list = f.read().splitlines()
        return _list

def getProductFrequency(lbls):
    labels = list(Counter(lbls).keys()) # equals to list(set(lbls))
    frequencies = list(Counter(lbls).values()) # counts the elements' frequency

    return [[labels[i], frequencies[i]] for i in range(0, len(labels))]

def writeOutputFile(resultGrocery):
    with open('grocery_output.txt', 'w+') as f:
        f.write("------------------------------------\n")
        f.write("Remaining Grocery\n")
        f.write("------------------------------------\n")
        for item in resultGrocery:
            f.write(item[0]+"\t"+item[1]+"\n")


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classes[classId] in finalClasses:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    # List of detected class labels
    detectLabels = [ classes[_id] for _id in classIds ]
    detectedProducts = getProductFrequency(detectLabels)
    return detectedProducts




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grocery Recognition using YOLOv3 in OpenCV')

    parser.add_argument('-w', '--weights',
		type=str,
		default='./model/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

    parser.add_argument('-cfg', '--config',
		type=str,
		default='./model/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-l', '--labels',
		type=str,
		default='./model/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

    parser.add_argument('-i', '--image',
		type=str,
		help='The path to the image file.')

    parser.add_argument('-v', '--video',
		type=str,
		help='The path to the video file.')

    parser.add_argument('-g', '--list',
		type=str,
		default='./grocery.txt',
		help='Path to the grocery list file.')

    args = parser.parse_args()

    # Initialize the parameters
    confThreshold = 0.8  # Confidence threshold
    nmsThreshold = 0.4   # Non-maximum suppression threshold
    inpWidth = 416       # Width of network's input image
    inpHeight = 416      # Height of network's input image

    # Get the class labels
    classes = getListfromFile(args.labels)

    # Filter class labels
    finalClasses = ['apple', 'orange']

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

    # Set DNN preferences
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the grocery list
    groceryList = getGroceryList(args.list)

    # Print the grocery list
    printGroceryList(groceryList)

    # Process inputs
    winName = 'Grocery Recognition'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

    if (args.image):
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)

    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)

    else:
        # Webcam input
        cap = cv2.VideoCapture(0)

    resultGrocery = []
    while cv2.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            writeOutputFile(resultGrocery)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Release device
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        detectedProducts = postprocess(frame, outs)

        # Remaining Grocery
        resultGrocery = getRemainingGrocery(groceryList, detectedProducts)
        printGroceryList(resultGrocery, title='Remaining Grocery')

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


        cv2.imshow(winName, frame)







