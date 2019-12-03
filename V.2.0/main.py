########################################################################
# Grocery Recognition
# CREATED BY: NEEL PATEL
# DATE CREATED: NOV 2019
########################################################################

# Usage example:  python main.py --list=grocery.txt --video=test.mp4
#                 python main.py --list=grocery.txt --image=test.jpg
#                 python main.py --list=grocery.txt

# Import standard packages
import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import sys
import argparse
from collections import Counter

# Import computer vision package
import cv2
# Import keras
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess


class YOLO(object):
    _defaults = {
        #"model_path": 'logs/trained_weights_final.h5',
        "model_path": 'model/trained_weights_final.h5',
        "anchors_path": 'model/yolo_anchors.txt',
        "classes_path": 'model/classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "text_size" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        #image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList


def getGroceryList(filename):
    groceryList = []
    g_list = getListfromFile(filename)
    
    for item in g_list:
        product, quantity = item.split('|')
        groceryList.append([product, quantity])

    return groceryList

def getRemainingGrocery(_groceryList, detectedProducts):
    result = _groceryList[:]
    for d_item in detectedProducts:
        for index, item in enumerate(result):
            # if label match
            if result[index][0] == d_item[0]:
                qty = int(result[index][1]) - d_item[1]
                if qty == 0:
                    result.remove(item)
                else:
                    result[index][1] = str(qty) # quanity

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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Grocery Recognition using YOLOv3 in OpenCV')

    parser.add_argument('-l', '--labels',
		type=str,
		default='./model/classes.txt',
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

    # Get the class labels
    classes = getListfromFile(args.labels)


    # Get the grocery list
    groceryList = getGroceryList(args.list)

    # Print the grocery list
    printGroceryList(groceryList)

    # Process inputs
    winName = 'Grocery Recognition'
    #cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

    yolo = YOLO()


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

            yolo.close_session()
            break

        # resize our captured frame if we need
        frame = cv2.resize(frame, (416,416), fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

        # detect object on our frame
        r_image, ObjectsList = yolo.detect_img(frame)

        if ObjectsList:
            # List of detected class labels
            detectLabels = [ (obj[6].split(' '))[0] for obj in ObjectsList ]
            detectedProducts = getProductFrequency(detectLabels)
            # Get the grocery list
            groceryList = getGroceryList(args.list)
            resultGrocery = getRemainingGrocery(groceryList, detectedProducts)

            printGroceryList(resultGrocery, title='Remaining Grocery')
        
       
        # show us frame with detection
        cv2.imshow(winName, r_image)

        

    




