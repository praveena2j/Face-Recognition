import numpy as np
import cv2 as cv
import os, sys, re, argparse, logging
from PIL import Image
from scipy.misc import imread

# Path Setup
module_list = ['/','/bob/eggs/bob.io.image-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.io.base-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.core-2.0.4-py2.7-linux-x86_64.egg', '/bob/eggs/bob.blitz-2.0.7-py2.7-linux-x86_64.egg', '/bob/eggs/bob.extension-2.0.8-py2.7.egg', '/bob/eggs/setuptools-18.3.2-py2.7.egg', '/bob/eggs/bob.learn.linear-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.learn.activation-2.0.3-py2.7-linux-x86_64.egg', '/bob/eggs/bob.math-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.flandmark-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.draw-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.color-2.0.3-py2.7-linux-x86_64.egg', '/bob/bin/']

cwd = os.getcwd()
for module in module_list:
    mod_path = cwd+'/..'+module
    sys.path.append(os.path.abspath(mod_path))

from bob.ip.flandmark import Flandmark
import cropfaces
from facedetect import utils
from facedetect import facedetect
import xml.etree.ElementTree as et

from sklearn import svm
from sklearn.svm import SVC

gUsage = "python recognize.py --video (-v) <Video Path> --trainset (-t) <Training XML>"

# Temporary files created, to be removed in the end
gTempFaceFile  = "Face.jpg"
gTempImageFile = "Image.jpg"

# For frontal face detection
gFrCascadePath = "haarcascade_frontalface.xml"
gfaceCount = 0 

def cleanTempFiles():
    utils.deleteFile(gTempFaceFile)
    utils.deleteFile(gTempImageFile)

def getConfigParams():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video")
    ap.add_argument("-t", "--trainset")
    args = vars(ap.parse_args())
    videoPath = args["video"]
    trainXML = args["trainset"]
    if not (videoPath or trainXML):
        print gUsage
        sys.exit(0);
    return videoPath, trainXML

def recognizeFace(aVideoPath, aTrainXML):
    # Read the test video and extract the faces
    recognizers = cv.createLBPHFaceRecognizer()
    recognizers.load(aTrainXML)
    LBP_hists = recognizers.getMatVector("histograms")
    
    # creating training data for modelling SVM
    training_data = []
    for i in range(len(LBP_hists)):
        training_data.append(LBP_hists[i][0])

    tree   = et.parse(aTrainXML)
    label_tags   = tree.find('labels')
    labels = label_tags.find('data').text
    clean_labels = re.sub(r"\s+"," ", labels)
    clean_labels = re.sub(r"(^\s+|\s+$)","", clean_labels)
    label_list = clean_labels.split(" ")

    # Initialize result array to 0
    result     = [0 for x in set(label_list)]
    result_svm = [0 for x in set(label_list)]
    
    index_tags = []
    index_tags = list(set(label_list))
    logging.info("Index Tags for Prof IDs: " + str(index_tags))
    # Training SVM
    clf = svm.LinearSVC()
    clf.fit(np.asarray(training_data), np.array(label_list))
    video = cv.VideoCapture(aVideoPath)
    frCC = cv.CascadeClassifier(gFrCascadePath)
    
    # keep looping over the frames in the video
    i = 1
    image = []
    global gfaceCount
    testlabel = []
    logging.info("SampleImageSize: 300x300" )
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break
        if i%10 == 0:
            skin = facedetect.getSkin(frame)
            faces = frCC.detectMultiScale(skin, 1.4, 10, 250)
            nbrt = 1
	    for (x,y,w,h) in faces:
                cv.rectangle(skin, (x,y),(x+w, y+h), (0,255,0), 2)
                Face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                localizer = Flandmark()
                keypoints = localizer.locate(Face, y, x, h, w)
                
        	if keypoints == None:
			continue
		left_x = (keypoints[5][1] + keypoints[1][1])/2
                left_y = (keypoints[5][0] + keypoints[1][0])/2
                right_x = (keypoints[6][1] + keypoints[2][1])/2
                right_y = (keypoints[6][0] + keypoints[2][0])/2
                
		cv.imwrite(gTempImageFile, frame)
                img = Image.open(gTempImageFile)
		cropfaces.CropFace(img, eye_left = (left_x, left_y), eye_right = (right_x, right_y), offset_pct =(0.3, 0.3), dest_sz = (300, 300)).save(gTempFaceFile)
		# Apply all the feature vectors on this face
                Res = imread(gTempFaceFile)
		Resimg = cv.cvtColor(Res, cv.COLOR_BGR2GRAY)
		
		image.append(Resimg)
		testlabel.append(nbrt)	
	     	recognizer = cv.createLBPHFaceRecognizer()
            	recognizer.train(image, np.array(testlabel))
            	LBP_Testhist = recognizer.getMatVector("histograms")
            	nbr_predicted_svm = clf.predict(LBP_Testhist[len(LBP_Testhist)-1])
                nbr_predicted, conf = recognizers.predict(Resimg)
                if conf<100:
                    logging.debug('\tnbr_predicted_svm: ' + str(nbr_predicted_svm[0]))
                    logging.debug('\tnbr_predicted: ' + str(nbr_predicted))
                    logging.debug('\tconf: ' + str(conf))
                    result[index_tags.index(str(nbr_predicted))] = result[index_tags.index(str(nbr_predicted))] + 1
		    result_svm[index_tags.index(nbr_predicted_svm[0])] = result_svm[index_tags.index(nbr_predicted_svm[0])] + 1
	    	    gfaceCount = gfaceCount + 1
		    if gfaceCount > 5:
                        logging.debug("Result: "+str(result))
                        logging.debug("Result SVM : "+str(result_svm))
			max_index = result.index(max(result))
    			max_index_svm = result_svm.index(max(result_svm))
    			logging.debug(videoPath +  ": Recognized professor ID: " + index_tags[max_index]  + " as per chi square")
    			logging.debug(videoPath +  ": Recognized professor ID: " + index_tags[max_index_svm]  + " as per SVM")
    			logging.info("********************* Complete *********************")
			# rec_image = cv.imread(str(index_tags[max_index])+".jpg")
   			# print rec_image
                        # cv.imshow("Recognizedface",rec_image)
   			# cv.waitKey(100)
		        return 0
        i = i + 1
    
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename = 'ProfessorRecognition.log', level = logging.DEBUG)
    videoPath, trainXML = getConfigParams()
    logging.info("Input Video: " + videoPath)
    logging.info("Input XML  : " + trainXML)
    if utils.existsFile(videoPath):
        recognizeFace(videoPath, trainXML)
    else:
	logging.error("No input video found")
    if gfaceCount < 5:
	logging.info("No professor Recognized. Faces Recognized: " + str(gfaceCount))
	logging.info("********************* Complete *********************")
    cleanTempFiles()
