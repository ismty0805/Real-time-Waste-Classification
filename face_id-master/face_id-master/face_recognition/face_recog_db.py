# face_recog.py

import face_recognition
import cv2
import camera
import os
import numpy as np
import sshtunnel
from pymongo import MongoClient, ReadPreference
from sshtunnel import SSHTunnelForwarder
import paramiko
from datetime import datetime, timedelta

class FaceRecog():
    
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = '/Users/q/Desktop/webcam/face_id-master/face_id-master/face_recognition/knowns'
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
'''
class WebCam():
    def __init__(self):
        self.classroom = 'N1_221'
        self.lecture = {'CS101':'2020:01:06:17:30:00', 'CS122':'2020:01:06:00:00:00', 'CS201':'2020:01:07:15:00:00'}    

    def get_difference(self, name):
        keys = list(self.lecture.keys())
        for key in keys:
            result = self.lecture[key].split(':')
            lecture_time_result = datetime(int(result[0]), int(result[1]), int(result[2]), int(result[3]), int(result[4]), int(result[5]))
            self.lecture[key] = lecture_time_result

        values = list(self.lecture.values())
        time = datetime.now()
        for value in values:
            value_att = value + datetime.timedelta(minutes = 90)
            print(value)
            print(time)
            print("---------------------------")
            difference = (value_att-time).seconds / 60
            print(difference)
            #if (difference )
            if (difference < 50.0 and difference > 20.0):
                val = value.strftime('%H:%M')
                print(val)
                col.update(
                {"$and":[{'student_id': name}, {'lecture_start_time' : val}]}, 
                {"$set": {'atd_check': 'Y'}}
                )
            else if (difference <= 20.0):
                val = value.strftime('%H:%M')
                print(val)
                col.update(
                {"$and":[{'student_id': name}, {'lecture_start_time' : val}]}, 
                {"$set": {'atd_check': 'L'}}
                )
            else:    
                break      
'''
if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    #webcam_id = WebCam()
    classroom = 'N1_112'
    lecture = {'CS496':['2020:01:01:15:00:00','2020:01:06:15:00:00','2020:01:08:15:00:00', '2020:01:13:15:00:00','2020:01:15:15:00:00', '2020:01:20:15:00:00', '2020:01:22:15:00:00','2020:01:27:15:00:00','2020:01:29:15:00:00']}
    values = list(lecture.values())

    for i in range(len(values)):
        for j in range(len(values[i])):
            print(values[i][j])
            result = values[i][j].split(':')
            lecture_time_result = datetime(int(result[0]), int(result[1]), int(result[2]), int(result[3]), int(result[4]), int(result[5]))
            values[i][j] = lecture_time_result
    values = list(lecture.values())
    time = datetime.now()

    while True:
        gets = face_recog.get_name()
        frame = face_recog.get_frame(gets)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # SSH통해서 mongodb접속하기 
        SSH_KEY_LOCATION = 'C:/Users/q/Downloads/cs496-key.pem' 
        JUMP_MACHINE_ADDRESS = '192.249.19.252'
        SSH_USER = 'root'
        REMOTE_MONGO_ADDRESS = '127.0.0.1'

        DB_NAME = 'attendance'
        COLLECTION_NAME = 'students'
        sshtunnel.SSH_TIMEOUT = 5.0
        sshtunnel.TUNNEL_TIMEOUT=5.0
        pkey = paramiko.RSAKey.from_private_key_file(SSH_KEY_LOCATION)
        server = SSHTunnelForwarder(
            (JUMP_MACHINE_ADDRESS, 2022),
            ssh_username=SSH_USER,
            ssh_private_key=pkey,
            remote_bind_address=(REMOTE_MONGO_ADDRESS, 27017),
            local_bind_address=('0.0.0.0', 27017)
        )

        # 서버 접속 시작 
        server.start()
        print(server.is_active, server.is_alive, server.tunnel_is_up)
        # mongodb 접속 
        client = MongoClient('mongodb://127.0.0.1:27017')
        db = client[DB_NAME]
        col = db[COLLECTION_NAME]
        docs = col.find_one({"userID":"yhs25"})
        print(docs)
        for get in gets :

            for value in values:
                for i in range(len(value)):
                    #value_att = value[i] + timedelta(minutes = 90)
                    print(value[i])
                    print(time)
                    print("---------------------------")
                    difference = (value[i]-time).total_seconds() / 60
                    print(difference)
                    if (difference >= 110.0):
                        day = value[i].strftime('%m/%d')
                        val = value[i].strftime('%H:%M')
                        print(val)
                        col.update(
                        {"$and":[{'student_id': get}, {'lecture_end_time' : val}, {'classroom':classroom}, {'day': day}]}, 
                        {"$set": {'atd_check': ''}}
                        )
                    elif (difference >= 80.0 and difference < 110.0):
                        day = value[i].strftime('%m/%d')
                        val = value[i].strftime('%H:%M')
                        print(val)
                        col.update(
                        {"$and":[{'student_id': get}, {'lecture_end_time' : val}, {'classroom':classroom}, {'atd_check':''}, {'day': day}]}, 
                        {"$set": {'atd_check': 'Y'}}
                        )
                    elif (difference > 60.0 and difference < 80.0):
                        day = value[i].strftime('%m/%d')
                        val = value[i].strftime('%H:%M')
                        print(val)
                        col.update(
                        {"$and":[{'student_id': get}, {'lecture_end_time' : val}, {'classroom':classroom}, {'atd_check':''}, {'day': day}]}, 
                        {"$set": {'atd_check': 'L'}}
                        )
                    elif (difference >= 0 and difference <= 60.0):
                        day = value[i].strftime('%m/%d')
                        val = value[i].strftime('%H:%M')
                        print(val)
                        col.update(
                        {"$and":[{'student_id': get}, {'lecture_end_time' : val}, {'classroom':classroom}, {'day': day}]}, 
                        {"$set": {'atd_check': 'N'}}
                        )
                    elif (difference < 0):
                        day = value[i].strftime('%m/%d')
                        val = value[i].strftime('%H:%M')
                        col.update(
                        {"$and":[{'student_id': get}, {'lecture_end_time' : val}, {'classroom':classroom}, {'atd_check':''}, {'day': day}]}, 
                        {"$set": {'atd_check': 'N'}}
                        )
                        continue   

            #webcam_id.get_difference(get) 

        docs = col.find()

        # collection 전체 출력 
        for i in docs:
            print(i)

        # 서버 접속 종료
        #server.stop()        

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            server.stop()
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
