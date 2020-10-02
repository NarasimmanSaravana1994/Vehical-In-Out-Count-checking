import cv2
import numpy as np
import time
from pyimagesearch.centroidtracker import CentroidTracker
#from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

cap=cv2.VideoCapture("/home/giuser/249_248_Source_backup/vehical_count/outpy.avi") #0 for 1st webcam
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time= time.time()
frame_id = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out_ = cv2.VideoWriter('outpy_testing.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

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
    centroid_checking = dict_[_id]
    if centroid_checking[0] > middle_cut_X and (centroid_checking[2][0] > OUT):
        error.append(1)
    elif (centroid_checking[2][0] > middle_cut_X) and (centroid_checking[2][1] > middle_cut_Y):
        IN_COUNT.append(1)
    elif (centroid_checking[2][0] < OUT) and (centroid_checking[2][0] < IN) and (centroid_checking[2][0] < OUT):
        OUT_COUNT.append(1)
    else:
        error.append(1)
    return True

#################### In out thresold logic #####################

#################### Setting up parameters ################

seconds = 1
fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds

#################### Initiate Process ################
h = None
w = None

while True:
    frameId = int(round(cap.get(1)))
    
    _,frame= cap.read() # 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_id+=1
    status = "Waiting"
    if w is None or h is None:
        (h, w) =  frame.shape[:2]
    if frameId % multiplier == 0:
        
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

                        #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                        #rectangle co-ordinaters
                        x=int(center_x - w/2)
                        y=int(center_y - h/2)
                        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                        boxes.append([x,y,w,h]) #put all rectangle areas
                        confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                        class_ids.append(class_id) #name of the object tha was detected


        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

        #  ROC lines 

        #cv2.line(frame, (573, 0), (573, 1096), (0, 255, 0), thickness=2)


        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(boxes)

        
        try:
        # loop over the tracked objects
            if objects is not None:
                for (objectID, centroid) in objects.items():

                    # in out logic
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

                    
                    
                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = ""
                    try:
                        dict_[objectID]
                        text = "ID : {} pos : {}".format(objectID,dict_[objectID][1])
                    except:
                        text = ""

                    # if boxes == []:
                    #     in_out_status(objectID,dict_)
                    #     del dict_[objectID]
                    #     ct.deregister(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                for i in range(len(boxes)):
                    if i in indexes:
                        x,y,w,h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence= confidences[i]
                        color = colors[class_ids[i]]
                        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                        cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
        except:
            ct = CentroidTracker()
                    
        elapsed_time = time.time() - starting_time
        fps = frame_id/elapsed_time
        cv2.putText(frame, "IN = {} & OUT = {}".format(len(IN_COUNT),len(OUT_COUNT)),(10, 50), font, 2, (0, 255, 0), 2)
        #cv2.putText(frame, "FPS :{} -- IN = {} & OUT = {}".format(str(round(fps, 2)),len(IN_COUNT),len(OUT_COUNT)),(10, 50), font, 0.5, (0, 255, 0), 2)

        cv2.imshow("Image", frame)
        out_.write(frame)

    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;
    
cap.release()    
cv2.destroyAllWindows()