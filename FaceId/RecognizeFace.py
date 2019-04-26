import ConfigParser
Config = ConfigParser.RawConfigParser()
import datetime
import mysql.connector
import commands
import numpy as np
import cv2 as cv
import os, sys, re, argparse, logging
from PIL import Image
from scipy.misc import imread

# Path Setup
module_list = ['/','/bob/eggs/bob.io.image-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.io.base-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.core-2.0.4-py2.7-linux-x86_64.egg', '/bob/eggs/bob.blitz-2.0.7-py2.7-linux-x86_64.egg', '/bob/eggs/bob.extension-2.0.8-py2.7.egg', '/bob/eggs/setuptools-18.3.2-py2.7.egg', '/bob/eggs/bob.learn.linear-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.learn.activation-2.0.3-py2.7-linux-x86_64.egg', '/bob/eggs/bob.math-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.flandmark-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.draw-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.color-2.0.3-py2.7-linux-x86_64.egg', '/bob/bin/']

cwd = os.getcwd()
for module in module_list:
    mod_path = cwd+module
    sys.path.append(os.path.abspath(mod_path))

from bob.ip.flandmark import Flandmark
import cropfaces
from facedetect import utils
from facedetect import facedetect
import xml.etree.ElementTree as et

from sklearn import svm
from sklearn.svm import SVC

gUsage = "python recognize.py --video (-v)"
gWorkDir = "/download1/professorIdentification/" 
gResultsDir = gWorkDir+"/data/"
gTrainedSetPath= gResultsDir+"trainingmodels/"
gLogFileName = gWorkDir + "ProfessorRecognition.log"
gCroppedFileName = gWorkDir + "recognize_input.mkv"
gLogConfigFileName = gWorkDir + "config.txt"
gTempFaceFile  = gWorkDir+"Face.jpg"
gTempImageFile = gWorkDir+"Image.jpg"
gFrCascadePath = "haarcascade_frontalface.xml"
gfaceCount = 0 
gTotalCount = 0
gMaxConfPercentage = -1

def createDir(aDirPath):
    # If directory does not exist create it
    if not os.path.exists(aDirPath):
        os.makedirs(aDirPath)

def cleanTempFiles():
    utils.deleteFile(gTempFaceFile)
    utils.deleteFile(gTempImageFile)

def getConfigParams():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video")
    args = vars(ap.parse_args())
    videoPath = args["video"]
    if not videoPath:
        print gUsage
        sys.exit(0);
    return videoPath

def recognizeFace(aVideoPath, aTrainXML,ttid):
    import pickle
    with open(aTrainXML, 'rb') as f:
        clf = pickle.load(f)
 
    commands.getstatusoutput('rm -f '+gCroppedFileName);
    commands.getstatusoutput('/opt/impartus/ffmpeg -y -ss 900 -i '+aVideoPath+'  -an -vcodec copy '+gCroppedFileName);
    video = cv.VideoCapture(gCroppedFileName)
    frCC = cv.CascadeClassifier(gFrCascadePath)
    
    # keep looping over the frames in the video
    i = 1
    #image = []
    #testlabel = []
    global gfaceCount
    global gTotalCount
    global gMaxConfPercentage
    global gfeaturevector
    isFaceFoundInPrev = False
    isCurrentFrameCropped = False
    windowX1 = 0
    windowY1 = 0
    windowX2 = 0
    windowY2 = 0
    next_frame = i 
    testimg = []
    window_started = False
    window_stopFrame = i
    k = 0
    gfeaturevector = []
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            logging.info("ttid="+str(ttid)+" Frame not grabbed")
            break
        if i == next_frame:
            height,width,channels = frame.shape
            scaleFactor = 1.4
            minNeighbors = 6
            isCurrentFrameCropped = False
            if isFaceFoundInPrev:
                frame = frame[windowY1:windowY2, windowX1:windowX2]
                isCurrentFrameCropped = True;

            isFaceFoundInPrev = False
            skin = facedetect.getSkin(frame)
            faces = frCC.detectMultiScale(skin, scaleFactor, minNeighbors, 250)
            nbrt = 1

            if len(faces)>1:
                # If multiple faces found in reference frame, go to next frame
                logging.info("ttid="+str(ttid)+ " checking for the next frame")
                i = i+1
                next_frame = i
                continue
       
            for (x,y,w,h) in faces:
                ##cv.rectangle(skin, (x,y),(x+w, y+h), (0,255,0), 2)
                Face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                localizer = Flandmark()
                keypoints = localizer.locate(Face, y, x, h, w)                    
                if (keypoints is None):
                    i = i+1
                    next_frame = i
                    continue
                left_x = (keypoints[5][1] + keypoints[1][1])/2
                left_y = (keypoints[5][0] + keypoints[1][0])/2
                right_x = (keypoints[6][1] + keypoints[2][1])/2
                right_y = (keypoints[6][0] + keypoints[2][0])/2

                cv.imwrite(gTempImageFile, frame)
                img = Image.open(gTempImageFile)
                cropfaces.CropFace(img, eye_left = (left_x, left_y), eye_right = (right_x, right_y), offset_pct =(0.2, 0.2), dest_sz = (200, 200)).save(gTempFaceFile)
        	    # Apply all the feature vectors on this face
                Res = cv.imread(gTempFaceFile)
                Resimg = cv.cvtColor(Res, cv.COLOR_BGR2GRAY)
                image =[]
                testlabel=[] 
    
                image.append(Resimg)
                testlabel.append(nbrt)	
             	recognizer = cv.createLBPHFaceRecognizer()
                recognizer.train(image, np.array(testlabel))
                LBP_Testhist = recognizer.getMatVector("histograms")
                nbr_predicted_svm = clf.predict(LBP_Testhist[len(LBP_Testhist)-1])
		
                if nbr_predicted_svm == 1:
                    if window_started == False:
                        logging.info("########## Window Started #########")
                        window_started = True
                        window_stopFrame = i + 50
                    gfaceCount = gfaceCount + 1
                    if isCurrentFrameCropped:
                        originalX = x + windowX1
                        originalY = y + windowY1
                    else:
                        originalX = x
                        originalY = y
                    windowX1 =  originalX-w if originalX-w > 0  else 0
                    windowY1 =  originalY-h if originalY-h > 0  else 0
                    windowX2 =  originalX+w+w if originalX+w+w < width else width
                    windowY2 =  originalY+h+h if originalY+h+h < height else height
                    isFaceFoundInPrev = True

                if window_started:
                    gfeaturevector.append(LBP_Testhist)
                    gTotalCount = gTotalCount + 1
                logging.info("ttid="+str(ttid)+", FrameNo="+str(i)+", NBR="+str(nbr_predicted_svm))

    	    if window_started:
                next_frame = i + 1
                if i > window_stopFrame:
                    logging.info("ttid="+str(ttid)+",########## Window Finished #########")
                    predicted_svm = []
                    if gTotalCount > 300:
                        median = np.median(gfeaturevector, axis = 0)
                        diff = np.sum((gfeaturevector - median)**2, axis=-1)
                        diff = np.sqrt(diff)
                        med_abs_deviation = np.mean(diff)
                        refinedfeature = []
                        modified_z_score = diff / med_abs_deviation
                        for count in range(len(gfeaturevector)):
                            if modified_z_score[count] < 1.0:
			                    refinedfeature.append(gfeaturevector[count])
                        logging.info("ttid="+str(ttid)+ " refined sample set size="+str(len(refinedfeature)))
                        output = np.squeeze(np.array(refinedfeature))
                        predicted_svm = clf.predict(list(output))
                        logging.info("ttid="+str(ttid)+" recognized 1's ="+str(np.count_nonzero(predicted_svm)))
               
                        Confidence_Score = (float(np.count_nonzero(predicted_svm)) / len(output))*100        
                        logging.info("ttid="+str(ttid)+", CONF="+str(Confidence_Score))
                        if Confidence_Score > gMaxConfPercentage:
                            gMaxConfPercentage = Confidence_Score
                        if Confidence_Score > 50:
                            return gMaxConfPercentage
                        gfaceCount = 0
                        gTotalCount = 0
                        gfeaturevector = []
                        refinedfeature = []
                    else:
                        logging.info("ttid="+str(ttid)+",Total facecount less than 300. TotalFaceCount="+str(gTotalCount))
                    window_started = False;	
    	    else:
    	        next_frame = i +10
        i = i + 1
    return gMaxConfPercentage


if __name__ == '__main__':
    startTime = datetime.datetime.now()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename = gLogFileName, level = logging.DEBUG)
    
    dataset = Config.read(gLogConfigFileName)
    if len(dataset) == 0:
	print "Failed to open config file "+gLogConfigFileName
	sys.exit(0)
    mysqlUser = Config.get('mysqldb', 'user')
    mysqlPassword = Config.get('mysqldb', 'password')
    mysqlHost = Config.get('mysqldb', 'host')
    mysqlDatabase = Config.get('mysqldb', 'database')

    videoPath = getConfigParams()
    logging.info("Input Video: " + videoPath)

    ttid = videoPath.split("/")[-1].split(".")[0]
    #conn = mysql.connector.connect(user='root', password='0$c!ent',host='10.8.0.201',database='oscient')
    conn = mysql.connector.connect(user=mysqlUser, password=mysqlPassword,host=mysqlHost,database=mysqlDatabase)
    cursor = conn.cursor()
    ##query = ("SELECT professorId,id FROM timetable where id="+ttid)
    query = ("SELECT tt.professorId, cl.name from timetable tt join classroom_m cl on tt.classroomid= cl.id where tt.id="+ttid)
    ##print query
    cursor.execute(query)
    for (professorId,classroomName) in cursor:
        uId = professorId
        clName = classroomName
        break
    cursor.close()
    conn.close()

    trainXML = gTrainedSetPath+str(uId)+".pkl";
    if utils.existsFile(videoPath):
        if utils.existsFile(trainXML):
            confPercentage =recognizeFace(videoPath, trainXML,ttid)
            endTime = datetime.datetime.now()
            diff = endTime - startTime
            logging.info("END ttid="+str(ttid)+"(ttid,profId,classRoom,TimeTaken,Confidence)"+" # "+str(ttid)+","+str(uId)+","+clName+","+str(diff)+","+str(confPercentage))
        else:
            logging.error("Train MODEL not found. xml="+trainXML)
    else:
	logging.error("No input video found")
    cleanTempFiles()
