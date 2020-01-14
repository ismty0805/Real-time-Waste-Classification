######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a webcam feed.
# It draws boxes, scores, and labels around the objects of interest in each frame
# from the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.
import face_recognition
from pymongo import MongoClient, ReadPreference
from sshtunnel import SSHTunnelForwarder
import paramiko
import camera
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from datetime import date
print("done import?")
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
print("import really done?")
# Name of the directory containing the object detection module we're using
MODEL_NAME = '/Users/q/Desktop/webcam/object_detection/inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()
print("got cwd")
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'label_map.pbtxt')
print("got all paths")
# Number of classes the object detector can identify
NUM_CLASSES = 3
print(PATH_TO_LABELS)
## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
print("got lb map")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
print("parsed")
category_index = label_map_util.create_category_index(categories)
print("starting to load model")
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

print("model loaded")
# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
today = date.today()
todate = today.strftime("%Y-%m-%d")
print(todate)
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
print("ok here")
# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

class VideoCamera(object):
    def __init__(self, video):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = video
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        return frame



class FaceRecog():
    
    def __init__(self, video):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = VideoCamera(video)

        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = '/Users/q/Desktop/webcam/object_detection/knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self, my_face_names = []):
        # Grab a single frame of video
        frame = self.camera.get_frame()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
            
        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, my_face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_name(self):

        if self.process_this_frame:
            
            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame
        
        return self.face_names


    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


plastic_cnt=0
metal_cnt=0
glass_cnt=0
foundface = None
face_recog = FaceRecog(video)
while(True):
    # print(1)
    gets = face_recog.get_name()
    frameface = face_recog.get_frame(gets)
    if foundface == None and len(gets)>0:
        foundface = str(gets[0])
        print(foundface)
    # show the frame
    # cv2.imshow("Frame", frameface)


    # SSH통해서 mongodb접속하기 
    SSH_KEY_LOCATION = 'C:/Users/q/Downloads/cs496-key.pem' 
    JUMP_MACHINE_ADDRESS = '192.249.19.252'
    SSH_USER = 'root'
    REMOTE_MONGO_ADDRESS = '127.0.0.1'
    DB_NAME = 'attendance'
    COLLECTION_NAME = 'students'

    pkey = paramiko.RSAKey.from_private_key_file(SSH_KEY_LOCATION)
    server = SSHTunnelForwarder(
        (JUMP_MACHINE_ADDRESS, 2022),
        ssh_username=SSH_USER,
        ssh_private_key=pkey,
        remote_bind_address=(REMOTE_MONGO_ADDRESS, 27017),
        local_bind_address=('0.0.0.0', 27017)
    )
    #  접속 시작 
    server.start()
    # print(server.is_active)
    DB_NAME = 'trash'
    COLLECTION_NAME = 'logins'
    # mongodb 접속 
    client = MongoClient('mongodb://127.0.0.1:27017')
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    # print(db, col)
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=6,
        min_score_thresh=0.95)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # print([category_index.get(i) for i in classes[0]])
    # print(scores)
    detected_object = category_index.get(classes[0][0])['name']
    score = scores[0][0]
    if score>0.95:
        if detected_object == 'glass':
            glass_cnt+=1
        elif detected_object == 'metal':
            metal_cnt+=1
        elif detected_object =='plastic':
            plastic_cnt+=1

    if glass_cnt>15:
        #send signal to arduino
        glass_cnt=0
        metal_cnt=0
        plastic_cnt=0
        if(not foundface==None):
            col.update_one({"userID": str(foundface)}, {"$push": {"points":{"type":"glass", "date":str(todate), "point":"20"}}})
            print("opened glass bin, "+str(foundface)+"+20pt!!")
            foundface=None
        else:
            print("opened glass bin")
    elif metal_cnt>15:
        #send signal to arduino
        glass_cnt=0
        metal_cnt=0
        plastic_cnt=0
        if(not foundface==None):
            col.update_one({"userID": str(foundface)}, {"$push": {"points":{"type":"metal", "date":str(todate), "point":"30"}}})
            print("opened metal bin, "+ str(foundface)+"+30pt!!")
            foundface=None
        else:
            print("opened metal bin")
    elif plastic_cnt>15:
        #send signal to arduino
        glass_cnt=0
        metal_cnt=0
        plastic_cnt=0
        if(not foundface==None):
            col.update_one({"userID": str(foundface)}, {"$push": {"points":{"type":"plastic", "date":str(todate), "point":"10"}}})
            print("opened plastic bin, "+ str(foundface)+"+10pt!!")
            foundface=None
        else:
            print("opened plastic bin")
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

