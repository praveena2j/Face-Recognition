import numpy as np
import cv2 as cv
import os, sys, argparse, logging
from PIL import Image
from scipy.misc import imread, imsave, imresize
from skimage.measure import structural_similarity as ssim
import math 
from matplotlib import pyplot

# Path Setup
module_list = ['/','/bob/eggs/bob.io.image-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.io.base-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.core-2.0.4-py2.7-linux-x86_64.egg', '/bob/eggs/bob.blitz-2.0.7-py2.7-linux-x86_64.egg', '/bob/eggs/bob.extension-2.0.8-py2.7.egg', '/bob/eggs/setuptools-18.3.2-py2.7.egg', '/bob/eggs/bob.learn.linear-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.learn.activation-2.0.3-py2.7-linux-x86_64.egg', '/bob/eggs/bob.math-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.flandmark-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.draw-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.color-2.0.3-py2.7-linux-x86_64.egg', '/bob/bin/']

cwd = os.getcwd()
for module in module_list:
    mod_path = cwd+'/..'+module
    sys.path.append(os.path.abspath(mod_path))

from bob.ip.flandmark import Flandmark
from bob.ip.draw import box, cross
import cropfaces
from facedetect import utils
from facedetect import facedetect
from matplotlib import pyplot
import xml.etree.ElementTree as ET

gUsage = "python FaceTraining.py --video <Video Path> --uid <User Id> --train 1|0"

# For profile face detetction
# gPrCascadePath = "haarcascade_profileface.xml"
# For frontal face detection
gFrCascadePath = "haarcascade_frontalface.xml"
geyeCascadePath = "frontalEyes35x16.xml"
gnamepair = {}
gnamepair = {1:"Prof.Rajeshwari", 2: "Prof.Arnold", 3:"Prof.Sachin", 4:"Prof.Ravindra", 5:"Prof.Deepa", 6:"Prof.Nandita", 7:"Ramachandra", 8:"Biiswas", 9:"Apoorva", 10:"Anil"}
# define the upper and lower boundaries
# of the HSV pixel intensities to be considered 'skin'
gLower = np.array([0, 48, 0], dtype = "uint8")
gUpper = np.array([100, 255, 255], dtype = "uint8")


gOutputDir    = 'OUTPUT'
gSamplePrefix = 'Sample'
gres = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
gres_SVM = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
    print videoPath, uId, trainFlag
    return videoPath, uId, trainFlag

def createSampleSet(aVideoPath, aUId):
    # Load the video	
    video = cv.VideoCapture(aVideoPath)
    # For face detection we will use the Haar Cascade provided by OpenCV.
    # prCC = cv.CascadeClassifier(gPrCascadePath)
    frCC = cv.CascadeClassifier(gFrCascadePath)
    eyeCC = cv.CascadeClassifier(geyeCascadePath)
    # Initialize variables
    i = 1
    faceCount = 1
        
    # keep looping over the frames in the video   
    while True:
        # Grab the current frame
        grabbed, frame = video.read()
        # Condition for end of the video
        if not grabbed:
            break

        # Looping over the frames  (one in 10 frames)
        if i%10 == 0:
            skin = getSkin(frame)
            faces = frCC.detectMultiScale(skin, 1.4, 10, 150) 
            
            print  "Found {0} faces!".format(len(faces))
            if not refFaceFound and len(faces)>0:
                if len(faces)>1:
                    # If multiple faces found in reference frame, go to next frame
                    print "checking for the next frame"
                    continue
                # Save the First Image as Reference
                (x,y,w,h) = faces[0]

                #filePath = saveFace(aUId, faceCount, frame[y:y+h,x:x+w])

            for (x,y,w,h) in faces:
                #faceImage = getFaceImage(skin, frame, im_size, x, y, w, h)
				gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                face_name = "Sample" + str(faceCount) + ".jpg"
				localizer = Flandmark()
				keypoints = localizer.locate(gray_image, y, x, h, w)
				print keypoints
				#print len(keypoints)
				if(keypoints) == None:
					continue	
				left_x = (keypoints[5][1] + keypoints[1][1])/2
				left_y = (keypoints[5][0] + keypoints[1][0])/2
				right_x = (keypoints[6][1] + keypoints[2][1])/2
				right_y = (keypoints[6][0] + keypoints[2][0])/2
				#cv.imwrite("Image.jpg", frame)
				#img = Image.open("Image.jpg")
		
				fileDir = os.path.join(gOutputDir, aUId)
				createDir(fileDir)
				filePath = os.path.join(fileDir, face_name)
		#print filePath
		cropfaces.CropFace(img, eye_left = (left_x, left_y), eye_right = (right_x, right_y), offset_pct =(0.2, 0.2), dest_sz = (200, 200)).save(filePath)
		Res = imread("Face.jpg")
                Resimg = cv.cvtColor(Res, cv.COLOR_BGR2GRAY)
		
                faceCount = faceCount+1
        i = i + 1
		
    # Cleanup the video and close any open windows
    video.release()
    cv.destroyAllWindows()


def getImagesAndLabels(aPath, aUId):
    # Append all the absolute image paths in a list image_path
    images = []
    # Labels will contains the label that is assigned to the image
    labels = []
    image_paths = [os.path.join(aPath, f) for f in os.listdir(aPath)]
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0])
        # append the images
        images.append(image)
        labels.append(aUId)
    # return the images list and labels list
    return images, labels

def saveFeatureVector(aUId):
    # For face recognition use LBPH Face Recognizer
    recognizer = cv.createLBPHFaceRecognizer()
    # Geet the face images and the corresponding labels of Sample Set
    path = os.path.join(gOutputDir, aUId)
    images, labels = getImagesAndLabels(path, aUId)

    # Perform the tranining
    recognizer.train(images, np.array(labels))
    hists = recognizer.getMatVector("histograms")

    # Save the feature vectors
    outputFileName = os.path.join(gOutputDir, aUId+'.xml') 
    recognizer.save(outputFileName)

# Apply the feature vector to recognize face
def loadFeatureVectors():
    recognizers = []
    if not os.path.exists(gOutputDir):
        logging.debug("Error: No training dir found")
    for filePath in os.listdir(gOutputDir):
        if filePath.endswith(".xml"):
            # For face recognition use LBPH Face Recognizer
            recognizer = cv.createLBPHFaceRecognizer()
            recognizer.load(os.path.join(gOutputDir, filePath))
            recognizers.append(recognizer)
           # print filePath
    print recognizers
    return recognizers

def recognizeFace(aVideoPath):
    # Read the test video and extract the faces
    recognizers = cv.createLBPHFaceRecognizer()
    #recognizers = loadFeatureVectors()
    recognizers.load(os.path.join(gOutputDir,'lbph_new.xml'))
    LBP_hists = recognizers.getMatVector("histograms")
    Training_data = []
    #print len(LBP_hists)
    #print type(LBP_hists)
    #print np.asarray(LBP_hists).shape

    for i in range(len(LBP_hists)):
        Training_data.append(LBP_hists[i][0])
    #print np.asarray(Training_data).shape

    # SVM for classification
    from sklearn import svm
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    
    tree = ET.parse(os.path.join(gOutputDir,'lbph_new.xml'))
    
    thingy = tree.find('labels')
    labels = thingy.find('data').text
    
    clf = svm.LinearSVC()
    clf.fit(np.asarray(Training_data), np.array(labels.split()))
    print aVideoPath
    video = cv.VideoCapture(aVideoPath)
    logging.debug('Loading the test video')
    frCC = cv.CascadeClassifier(gFrCascadePath)
    
    # keep looping over the frames in the video
    i = 1
    gFaceCount = 1
    image = []
    testlabel = []
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break
        if i%10 == 0:
            skin = getSkin(frame)
            logging.debug('skin detection was successfully completed')
            faces = frCC.detectMultiScale(skin, 1.4, 10, 150)
            logging.debug('Face detection was successfully completed')
            nbrt = 1
	    for (x,y,w,h) in faces:
                cv.rectangle(skin, (x,y),(x+w, y+h), (0,255,0), 2)
                Face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                localizer = Flandmark()
                keypoints = localizer.locate(Face, y, x, h, w)
                
		if keypoints == None:
			print "No keypoints found"
			continue
		left_x = (keypoints[5][1] + keypoints[1][1])/2
                left_y = (keypoints[5][0] + keypoints[1][0])/2
                right_x = (keypoints[6][1] + keypoints[2][1])/2
                right_y = (keypoints[6][0] + keypoints[2][0])/2
                
		cv.imwrite("Image.jpg", frame)
                img = Image.open("Image.jpg")
		cropfaces.CropFace(img, eye_left = (left_x, left_y), eye_right = (right_x, right_y), offset_pct =(0.3, 0.3), dest_sz = (200, 200)).save("Face.jpg")
		# Apply all the feature vectors on this face
		
                Res = imread("Face.jpg")
		Resimg = cv.cvtColor(Res, cv.COLOR_BGR2GRAY)
		
		image.append(Resimg)
		testlabel.append(nbrt)	
	     	recognizer = cv.createLBPHFaceRecognizer()
            	recognizer.train(image, np.array(testlabel))
            	LBP_Testhist = recognizer.getMatVector("histograms")
		
		# print(len(LBP_Testhist))
                # print type(LBP_Testhist)
                # print np.asarray(LBP_Testhist).shape

            	nbr_predicted_SVM = clf.predict(LBP_Testhist[len(LBP_Testhist)-1])
            	# print int(nbr_predicted_SVM[0])
		#gres_SVM[int(nbr_predicted_SVM[0])-1] = gres_SVM[int(nbr_predicted_SVM[0])-1] + 1
		#if gFaceCount > 5:
		#    print aVideoPath + "is reconized to" + 
                #    return 0
		#gFaceCount = 1
                nbr_predicted, conf = recognizers.predict(Resimg)
                if conf<100:
		    print int(nbr_predicted_SVM[0])
                    print nbr_predicted, conf
		    cv.imshow(gnamepair[nbr_predicted], frame[y:y+h, x:x+w])
                    cv.waitKey(100)
		    gres[nbr_predicted-1] = gres[nbr_predicted-1] + 1
		    gres_SVM[int(nbr_predicted_SVM[0])-1] = gres_SVM[int(nbr_predicted_SVM[0])-1] + 1
	    	    gFaceCount = gFaceCount + 1
       		    # print gFaceCount
		    if gFaceCount > 5:
		        return 0
        i = i + 1
    logging.info('Face Recognition was successfully completed')

if __name__ == '__main__':
    logging.basicConfig(filename = 'ProfessorIdentification.log', level = logging.DEBUG)
    videoPath, uid, training_flag = getConfigParams()
    if training_flag == "1":
        print "Training Data for Professor", uid
        createDir(gOutputDir)
        createSampleSet(videoPath, uid)
        saveFeatureVector(uid)
    else:
        print "Applying Data for Professor"
        recognizeFace(videoPath)
	max_value = max(gres)
	#print max_value
	max_value_SVM = max(gres_SVM)
        max_index = gres.index(max_value)
	#print max_value_SVM
	max_index_SVM = gres_SVM.index(max_value_SVM)
	Prof =  cv.imread(str(max_index+1)+".jpg")
        cv.imshow(gnamepair[max_index+1], Prof)
        cv.waitKey(500000)
	f = open('output.txt', 'a')
	f.write( videoPath +  "is the recognized professor to" + str(max_index+1)  + " as per chi square\n")
	f.write( videoPath +  "is the recognized professor to" + str(max_index_SVM+1)  + " as per SVM\n")
	f.close()
        # print str(max_index+1) + "is the recognized professor as per chi square"  
	# print str(max_index_SVM+1) + "is the recognized professor as per SVM"
        print "Done"
