import commands
import numpy as np
import cv2 as cv
import os, sys, argparse, logging
from PIL import Image
from scipy.misc import imread, imsave, imresize
import math 
from matplotlib import pyplot
import time

import re
from operator import itemgetter

from sklearn import svm
from sklearn.svm import SVC

# Path Setup
module_list = ['/','/bob/eggs/bob.io.image-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.io.base-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.core-2.0.4-py2.7-linux-x86_64.egg', '/bob/eggs/bob.blitz-2.0.7-py2.7-linux-x86_64.egg', '/bob/eggs/bob.extension-2.0.8-py2.7.egg', '/bob/eggs/setuptools-18.3.2-py2.7.egg', '/bob/eggs/bob.learn.linear-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.learn.activation-2.0.3-py2.7-linux-x86_64.egg', '/bob/eggs/bob.math-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.flandmark-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.draw-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.color-2.0.3-py2.7-linux-x86_64.egg', '/bob/bin/']

cwd = os.getcwd()
for module in module_list:
    mod_path = cwd + module
    sys.path.append(os.path.abspath(mod_path))

import pickle
from sklearn.externals import joblib
from bob.ip.flandmark import Flandmark
from bob.ip.draw import box, cross
import cropfaces
from facedetect import utils
from facedetect import facedetect
from matplotlib import pyplot
import xml.etree.ElementTree as ET

gUsage = "python FaceTraining.py --video <Video Path> --uid <User Id> --train 1|2|3"

gFrCascadePath = "haarcascade_frontalface.xml"
geyeCascadePath = "frontalEyes35x16.xml"
# define the upper and lower boundaries
# of the HSV pixel intensities to be considered 'skin'
gLower = np.array([0, 48, 0], dtype = "uint8")
gUpper = np.array([100, 255, 255], dtype = "uint8")
gWorkDir = "/download1/professorIdentification/"
gOutputDir = gWorkDir + '/data/'
gFacesDir = gOutputDir + '/faces/'
gRefinedDir = gOutputDir + '/refined/'
gTraingModelsDir = gOutputDir + '/trainingmodels/'
gLogFileName = gWorkDir + 'ProfessorTraining.log'
gTrainingFile = gWorkDir + 'training_input.mp4'

gres = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
gres_SVM = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
images = []
labels = []

def createDir(aDirPath):
    # If directory does not exist create it
    if not os.path.exists(aDirPath):
        os.makedirs(aDirPath)

def getConfigParams():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video")
    ap.add_argument("-u", "--uid")
    ap.add_argument("-t", "--train")
    args = vars(ap.parse_args())
    videoPath = args["video"]
    uId = args["uid"]
    trainFlag = args["train"]
    if not videoPath or (trainFlag and not uId):
        print gUsage
        sys.exit(0);
    ##print videoPath, uId, trainFlag
    return videoPath, uId, trainFlag

def createSampleSet(aVideoPath, aUId,ttid):
    # Load the video
    commands.getstatusoutput('rm '+gTrainingFile);
    commands.getstatusoutput('/opt/impartus/ffmpeg -y -ss 900 -i "'+aVideoPath+'" -an -vcodec copy '+gTrainingFile);	
    ##commands.getstatusoutput('ffmpeg -y -i '+aVideoPath+' -an -vcodec copy input.mp4');	
    video = cv.VideoCapture(gTrainingFile)
    # For face detection we will use the Haar Cascade provided by OpenCV.
    # prCC = cv.CascadeClassifier(gPrCascadePath)
    frCC = cv.CascadeClassifier(gFrCascadePath)
    eyeCC = cv.CascadeClassifier(geyeCascadePath)
    # Initialize variables
    i = 1
    faceCount = 1
    refFaceFound = 0
    # keep looping over the frames in the video 
    #video.set(CV_CAP_PROP_POS_FRAMES,25*10*60)  
    while True:
        # Grab the current frame
        grabbed, frame = video.read()
        # Condition for end of the video
        if not grabbed:
            break
        # Looping over the frames  (one in 10 frames)
        if i%10 == 0:
            skin = getSkin(frame)
            faces = frCC.detectMultiScale(skin, 1.4, 10, 250) 
            
            ##loggin.info("ttid="+str(ttid)+" Found {0} faces!".format(len(faces)))
            logging.info("ttid="+str(ttid)+" Found "+str(len(faces))+" faces! frameNo="+str(i))
            if not refFaceFound and len(faces)>0:
                if len(faces)>1:
                    # If multiple faces found in reference frame, go to next frame
                    logging.info("ttid="+str(ttid)+" Checking for the next frame");
                    continue
                # Save the First Image as Reference
                (x,y,w,h) = faces[0]

            for (x,y,w,h) in faces:
                #faceImage = getFaceImage(skin, frame, im_size, x, y, w, h)
                gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ts = "{:.3f}".format(time.time()).replace(".","")
               
                face_name = aUId +"_" + str(ts) + ".jpg"
                localizer = Flandmark()
                keypoints = localizer.locate(gray_image, y, x, h, w)
              
                if(keypoints) == None:
                    continue
                left_x = (keypoints[5][1] + keypoints[1][1])/2
                left_y = (keypoints[5][0] + keypoints[1][0])/2
                right_x = (keypoints[6][1] + keypoints[2][1])/2
                right_y = (keypoints[6][0] + keypoints[2][0])/2
		       
                frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                
                fileDir = os.path.join(gFacesDir,aUId)
                createDir(fileDir)
                filePath = os.path.join(fileDir, face_name)
                logging.info("ttid="+str(ttid)+". FilePath="+filePath)
                cropfaces.CropFace(img, eye_left = (left_x, left_y), eye_right = (right_x, right_y), offset_pct =(0.2, 0.2), dest_sz = (200,200)).save(filePath)
                
                faceCount = faceCount+1
        i = i + 1
    # Cleanup the video and close any open windows
    video.release()
    cv.destroyAllWindows()


def getImagesAndLabelsforrefinement(aPath):
    # Images will contains face images
    images_ = []
    # Labels will contains the label that is assigned to the image
    labels_ = []
    # Append all the absolute image paths in a list image_path
    imagePaths = [os.path.join(aPath, filePath) for filePath in os.listdir(aPath)]
    for imagePath in imagePaths:
        # Read the image and convert to grayscale
        imagePil = Image.open(imagePath).convert('L')
        # Convert the image format into numpy array
        image = np.array(imagePil, 'uint8')
        # append the images
        images_.append(image)
        label = imagePath.split("/")[-1].split(".")[0].split("_")[1]
        labels_.append(int(label))
    return images_, labels_

def getImagesAndLabels(aPath, aUId):
    # Images will contains face images
    #images = []
    # Labels will contains the label that is assigned to the image
    #labels = []
    # Append all the absolute image paths in a list image_path
    print len(images)
    imagePaths = [os.path.join(aPath, filePath) for filePath in os.listdir(aPath)]
    for imagePath in imagePaths:
        # Read the image and convert to grayscale
        #print imagePath
        imagePil = Image.open(imagePath).convert('L')
        # Convert the image format into numpy array
        image = np.array(imagePil, 'uint8')
        # append the images
        images.append(image)
        label = imagePath.split("/")[-1].split(".")[0].split("_")[1]
        #labels.append(int(label))
        labels.append(int(aUId))
    return images, labels

def refineFaces(aUId):
    path = os.path.join(gFacesDir, aUId)
    images, labels = getImagesAndLabelsforrefinement(path)

    if len(images) < 1000:
        print str(aUId)+"data set less than 1000"
        return 0
    # Perform the tranining
    recognizer = cv.createLBPHFaceRecognizer()
    recognizer.train(images, np.array(labels))
    hists = recognizer.getMatVector("histograms")
    median = np.median(hists, axis = 0)
    diff = np.sum((hists - median)**2, axis=-1)
    diff = np.sqrt(diff)
    indices, L_sorted = zip(*sorted(enumerate(diff), key=itemgetter(1)))

    refinedindex = list(indices)
    refinedlist = refinedindex[0:500]
    refined_list = []

    for i in refinedlist:
        refined_list.append(labels[i])
    createDir(gRefinedDir+aUId)
    for fil in refined_list:
        img = cv.imread(os.path.join(gFacesDir, aUId)+"/" +aUId+"_"+ str(fil) + '.jpg')
        if img is not None:
            cv.imwrite(os.path.join(gRefinedDir, aUId)+"/"+aUId+"_"+str(fil) + '.jpg', img)

def saveTrainingModels():
    # For face recognition use LBPH Face Recognizer
    recognizer = cv.createLBPHFaceRecognizer()
    # Geet the face images and the corresponding labels of Sample Set
    x = os.listdir(os.path.join(gRefinedDir))
    createDir(gTraingModelsDir)
    print x
    for profid in x:
        images = []
        labels = []
        print profid
        for i in x:
            path = os.path.join(gRefinedDir ,i)
            #print path
            imagePaths = [os.path.join(path, filePath) for filePath in os.listdir(path)]
            if i == profid:
                lab = '1'
            else:
                lab = '0'
            for imagePath in imagePaths:
                # Read the image and convert to grayscale
                imagePil = Image.open(imagePath).convert('L')
                # Convert the image format into numpy array
                image = np.array(imagePil, 'uint8')
                # append the images
                images.append(image)
                labels.append(int(lab))
            print len(images)
        # Perform the tranining
        recognizer.train(images, np.array(labels))
        hists = recognizer.getMatVector("histograms")

        training_data = []
        for i in range(len(hists)):
            training_data.append(hists[i][0])

        clf = svm.LinearSVC()
        clf.fit(np.asarray(training_data), np.array(labels))

        with open(gTraingModelsDir + profid + '.pkl', 'wb') as f:
            pickle.dump(clf, f)
    print "Done"


def getSkin(aFrame):
    converted = cv.cvtColor(aFrame, cv.COLOR_BGR2HSV)
    # Determine the HSV pixel intensities that fall into the speicifed upper and lower boundaries
    skinMask = cv.inRange(converted, gLower, gUpper)
    # Apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    skinMask = cv.dilate(skinMask, kernel, iterations = 6)
    # Blur the mask to help remove noise, then apply the mask to the frame
    skinMask = cv.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv.bitwise_and(aFrame, aFrame, mask = skinMask)
    return skin

if __name__ == '__main__':
    ##logging.basicConfig(filename = 'ProfessorIdentification.log', level = logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename = gLogFileName, level = logging.DEBUG)
    videoPath, uid, training_flag = getConfigParams()
    logging.debug("--video "+videoPath+" --uid "+str(uid)+" --train "+str(training_flag))
    if training_flag == "1":
        ttid = videoPath.split("=")[1].split("&")[0]
        logging.info("ttid="+str(ttid)+" Training Data for Professor uid="+str(uid))
        createDir(gOutputDir)
        createSampleSet(videoPath, uid,ttid)
        logging.info("ttid="+str(ttid)+" Sample Set Creation Done. uid="+str(uid))
    elif training_flag == "2":
	    refineFaces(uid)
    elif training_flag == '3':
        saveTrainingModels()
    else:
        logging.info("ERROR: Invalid training value")
