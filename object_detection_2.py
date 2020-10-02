import cv2
import numpy as np
import time
from pyimagesearch.centroidtracker import CentroidTracker
#from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
import dlib
#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3
#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

cap=cv2.VideoCapture("/home/giuser/249_248_Source_backup/vehical_count/ref/People-Counting-in-Real-Time/outpy.avi") #0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0

ct = CentroidTracker()
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
x_ = []
empty=[]
empty1=[]

#################### Setting up parameters ################

seconds = 1
fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds

#################### Initiate Process ################
H = None
W = None
count = 0
while True:
    
    frameId = int(round(cap.get(1)))
    
    _,frame= cap.read() # 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_id+=1
    status = "Waiting"
    rects = []
    if W is None or H is None:
        (H, W) = (1800, 1100)
    # if w is None or h is None:
    #     (h, w) =  (1800, 1100)#frame.shape[:2]
    if frameId % multiplier == 0:
        
        height,width,channels = frame.shape

        #height , width , layers =  frame.shape
        new_h=height/2
        new_w=width/2
        frame = cv2.resize(frame, (1800, 1100))
        # cv2.imwrite("test.jpg", frame)
        # break
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
                i = count
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id != 0:
                    
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

                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(x, y, w, h)
                        tracker.start_track(rgb, rect)
                        trackers.append(tracker)

    else:
			# loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
    cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)
        #indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

        # loop over the tracked objects
    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            status = "Tracking"
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < h // 2:
                    totalUp += 1
                    empty.append(totalUp)
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > h // 2:
                    totalDown += 1
                    empty1.append(totalDown)
                    #print(empty1[-1])
                    x_ = []
                    # compute the sum of total people inside
                    x_.append(len(empty1)-len(empty))
                    #print("Total people inside:", x)
                    # Optimise number below: 10, 50, 100, etc., indicate the max. people inside limit
                    # if the limit exceeds, send an email alert
                    # people_limit = 10
                    # if sum(x_) == people_limit:
                    #     if ALERT:
                    #         print("[INFO] Sending email alert..")
                    #         #Mailer().send(MAIL)
                    #         print("[INFO] Alert sent")

                    to.counted = True


        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
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
    # construct a tuple of information we will be displaying on the
    info = [
        ("Exit", totalUp),
        ("Enter", totalDown),
        ("Status", status),
    ]

    info2 = [
        ("Total people inside", x_),
    ]

    # Display the output
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for (i, (k, v)) in enumerate(info2):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (265, H - ((i * 20) + 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elapsed_time = time.time() - starting_time
    fps = frame_id/elapsed_time
    cv2.putText(frame, "FPS:"+str(round(fps, 2)),
                (10, 50), font, 2, (0, 0, 0), 1)

    cv2.imshow("Image", frame)
    count = count +1

    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;
    
cap.release()    
cv2.destroyAllWindows()
