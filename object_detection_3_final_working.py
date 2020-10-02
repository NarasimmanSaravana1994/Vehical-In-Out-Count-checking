import cv2
import csv
import numpy as np
import time
import datetime
from datetime import date
from pyimagesearch.centroidtracker import CentroidTracker
#from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
import imutils
import pyodbc
#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

#cap=cv2.VideoCapture("/home/giuser/249_248_Source_backup/vehical_count/outpy.avi") #0 for 1st webcam
cap = video_capture = cv2.VideoCapture("rtsp://admin:gi@12345@192.168.15.55:554/Streaming/Channels/101")
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time= time.time()
frame_id = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out_ = cv2.VideoWriter(str(date.today())+'_output_.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (1200, 678))

ct = CentroidTracker()
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
x_ = []
empty=[]
empty1=[]

#################### In out thresold logic #####################
OUT = 320 # old 340
IN = 580
middle_cut_X = 575
middle_cut_Y = 150
middle_id_creation = 590

dict_ = {}

IN_COUNT = []
OUT_COUNT = []
error = []


def in_out_status(_id,dict_):
    try:
        connectionString = 'DRIVER={FreeTDS};SERVER=10.101.1.190\SQL2014;DATABASE=InventoryDB30072019;UID=sa;PWD=gitech123*gitech;'
        connection = pyodbc.connect(connectionString, autocommit=True)
        cursor = connection.cursor()

        now = datetime.datetime.now()
        current_date = now.strftime("%Y_%m_%d")
        centroid_checking = dict_[_id]
        if (centroid_checking[0] > middle_cut_X) and (centroid_checking[2][0] > OUT):
            error.append(1)
        elif (centroid_checking[2][0] > middle_cut_X) and (centroid_checking[2][1] > middle_cut_Y):
            IN_COUNT.append(1)
            print("comming inside  ==  ",str(datetime.datetime.now()))
            cursor.execute("TRA_INS_VehicleInOutTrack ?,?,?", [1,0,str(datetime.datetime.now())])
            connection.commit()
            connection.close()
        elif (centroid_checking[2][0] < OUT) and (centroid_checking[2][0] < IN) and (centroid_checking[2][0] < OUT) and (centroid_checking[2][1]>150) and (centroid_checking[0]>150):
            OUT_COUNT.append(1)
            print("comming outside  ==  ",str(datetime.datetime.now()))
            cursor.execute("TRA_INS_VehicleInOutTrack ?,?,?", [0,1,str(datetime.datetime.now())])
            connection.commit()
            connection.close()
        else:
            error.append(1)
    except Exception as ex:
        print("#################  = ",ex)
        return False


    # if changed:
    #     print("comming inside...")
    #     with open(current_date+"_live_in_out_count.csv", mode='a') as csv_file:
    #         fieldnames = ['IN', 'OUT','TIME']
    #         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #         writer.writerow({'IN': str(len(IN_COUNT)), 'OUT': str(len(OUT_COUNT)),'TIME':str(datetime.datetime.now())})
        
    return True

#################### In out thresold logic #####################

#################### Setting up parameters ################

seconds = 1
fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds

#################### Initiate Process ################
h = None
w = None
previous_obj_ids =[]
frame_count = 0
while True:
    try:
    
        frameId = int(round(cap.get(1)))
        
        try:
            _,frame= cap.read() # 
            frame = imutils.resize(frame, width=1200)
        except Exception as ex:
            print("Frame === ",frame)
            print("###################################")
            cap = video_capture = cv2.VideoCapture("rtsp://admin:gi@12345@192.168.15.55:554/Streaming/Channels/101")
            _,frame= cap.read() # 
            frame = imutils.resize(frame, width=1200)
        
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count+=1
        status = "Waiting"
        if w is None or h is None:
            (h, w) =  frame.shape[:2]
        
        if frameId % multiplier == 0:
        #if frame_count == 20:
            frame_count =0
            
            height,width,channels = frame.shape
            #detecting objects
            blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    

                
            net.setInput(blob)
            outs = net.forward(outputlayers)
            #print(outs[1])


            #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
            class_ids=[]
            confidences=[]
            boxes=[]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    #if class_id != 0 and class_id != 2 and class_id != 3:
                    if class_id == 7:
                        
                        confidence = scores[class_id]
                        if confidence > 0.7:
                            status = "Detecting"
                            trackers = []
                            #onject detected
                            center_x= int(detection[0]*width)
                            center_y= int(detection[1]*height)
                            w = int(detection[2]*width)
                            h = int(detection[3]*height)

                            cv2.circle(frame,(center_x,center_y),10,(255,255,0),2)
                            #rectangle co-ordinaters
                            x=int(center_x - w/2)
                            y=int(center_y - h/2)
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                            boxes.append([x,y,w+x,y+h]) #put all rectangle areas
                            confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                            class_ids.append(class_id) #name of the object tha was detected


            indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

            #  ROC lines 

            #cv2.line(frame, (573, 0), (573, 1096), (0, 255, 0), thickness=2)


            # update our centroid tracker using the computed set of bounding
            # box rectangles
            objects = ct.update(boxes)
            current_obj_ids=[]
            
            # loop over the tracked objects
            if objects is not None:   
                for (objectID, centroid) in objects.items():
                    
                    

                    # 1.compare find the missing id
                    # 2.check in/out logic and send update
                    # 3.remove id in custome dictionary
                    # 4. update previous dic

                    #compare logic
                    current_obj_ids.append(objectID)
                    current_position = ""
                    try:
                        dict_[objectID]

                        if centroid[0] > OUT and (centroid[0] < IN):
                            current_position = "middle postion"
                        elif centroid[0] < OUT and centroid[0] < IN:
                            current_position = "initial position"
                        elif centroid[0] < IN and centroid[0] < OUT:
                            current_position = "last position"

                        current_coordinates = centroid

                        dict_[objectID][1] = current_position
                        dict_[objectID][2] = current_coordinates
                    except Exception as ex:
                        #print(ex)
                        if centroid[0] > OUT and (centroid[0] < IN):
                            current_position = "middle postion"
                        elif centroid[0] < OUT and centroid[0] < IN:
                            current_position = "initial position"
                        elif centroid[0] < IN and centroid[0] < OUT:
                            current_position = "last position"

                        current_coordinates = centroid

                        dict_[objectID] = [centroid[0],current_position,current_coordinates]

                    # in out logic
                    text = ""
                    try:
                        dict_[objectID]
                        text = "ID : {} pos : {}".format(objectID,dict_[objectID][1])
                    except:
                        text = ""

                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        
            destroyed_obj_ids=set(previous_obj_ids).difference(current_obj_ids)
            #print(destroyed_obj_ids)
            # 
            for _id in destroyed_obj_ids:
                in_out_status(_id,dict_)
                del dict_[_id]
            previous_obj_ids = current_obj_ids
            elapsed_time = time.time() - starting_time
            fps = frame_id/elapsed_time
            cv2.putText(frame, "IN = {} & OUT = {}".format(len(IN_COUNT),len(OUT_COUNT)),(10, 50), font, 2, (0, 255, 0), 2)
            #cv2.putText(frame, "FPS :{} -- IN = {} & OUT = {}".format(str(round(fps, 2)),len(IN_COUNT),len(OUT_COUNT)),(10, 50), font, 0.5, (0, 255, 0), 2)

            #cv2.imshow("Image", frame)
            out_.write(frame)

        key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
        
        if key == 27: #esc key stops the process
            break;
    except Exception as ex:
        print("Error occured == ",ex)
        cap = video_capture = cv2.VideoCapture("rtsp://admin:gi@12345@192.168.15.55:554/Streaming/Channels/101")
        pass
    
cap.release()    
cv2.destroyAllWindows()