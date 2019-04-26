from pyimagesearch import imutils
import numpy as np
import cv2 as cv
import os, sys, argparse, logging
from PIL import Image
from scipy.misc import imread, imsave, imresize
from skimage.measure import structural_similarity as ssim
import math 
from matplotlib import pyplot

# Path Setup
module_list = ['/bob/eggs/bob.io.image-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.io.base-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.core-2.0.4-py2.7-linux-x86_64.egg', '/bob/eggs/bob.blitz-2.0.7-py2.7-linux-x86_64.egg', '/bob/eggs/bob.extension-2.0.8-py2.7.egg', '/bob/eggs/setuptools-18.3.2-py2.7.egg', '/bob/eggs/bob.learn.linear-2.0.6-py2.7-linux-x86_64.egg', '/bob/eggs/bob.learn.activation-2.0.3-py2.7-linux-x86_64.egg', '/bob/eggs/bob.math-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.flandmark-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.draw-2.0.2-py2.7-linux-x86_64.egg', '/bob/eggs/bob.ip.color-2.0.3-py2.7-linux-x86_64.egg', '/bob/bin/']

cwd = os.getcwd()
for module in module_list:
    sys.path.append(cwd+ module)

from bob.ip.flandmark import Flandmark
from bob.ip.draw import box, cross
import cropfaces

gUsage = "python FaceTraining.py --video <Video Path> --uid <User Id> --train 1|0"

# For profile face detetction
# gPrCascadePath = "haarcascade_profileface.xml"
# For frontal face detection
gFrCascadePath = "haarcascade_frontalface.xml"
gnamepair = {}
gnamepair = {1:"Prof.Rajeshwari", 2: "Prof.Swathi", 3:"Prof.Sachin", 4:"Prof.Ravindra", 5:"Prof.Deepa", 6:"Prof.Nandita"}
# define the upper and lower boundaries
# of the HSV pixel intensities to be considered 'skin'
gLower = np.array([0, 48, 0], dtype = "uint8")
gUpper = np.array([100, 255, 255], dtype = "uint8")

gOutputDir    = 'output'
gSamplePrefix = 'Sample'
gres = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
    print videoPath, uId, trainFlag
    return videoPath, uId, trainFlag

def saveFace(aUId, aFaceCount, aFace):
    fileName = gSamplePrefix + str(aFaceCount)+ ".png"
    fileDir = os.path.join(gOutputDir, aUId)
    createDir(fileDir)
    filePath = os.path.join(fileDir, fileName)
    cv.imwrite(filePath, aFace)
    return filePath

def getFaceImage(aSkin, aFrame, aRefSize, aX, aY, aW, aH):
    cv.rectangle(aSkin, (aX,aY),(aX+aW, aY+aH), (0,255,0), 2)
    tmpImage = imresize((aFrame[aY:aY+aH,aX:aX+aW]),(aRefSize[0], aRefSize[1]))
    faceImage = cv.cvtColor(tmpImage, cv.COLOR_BGR2GRAY)
    return faceImage

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

def getRefImage(aFilePath):
    image_Ref_face = imread(aFilePath)
    imSize = image_Ref_face.shape
    trainImage = cv.cvtColor(image_Ref_face, cv.COLOR_BGR2GRAY)
    return imSize, trainImage

def createSampleSet(aVideoPath, aUId):
    # Load the video	
    video = cv.VideoCapture(aVideoPath)
    # For face detection we will use the Haar Cascade provided by OpenCV.
    # prCC = cv.CascadeClassifier(gPrCascadePath)
    frCC = cv.CascadeClassifier(gFrCascadePath)

    # Initialize variables
    i = 1
    faceCount = 1
    refFaceFound = 0
    
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
            faces = frCC.detectMultiScale(skin, 1.4, 10)
            #faces = detect(skin, frCC)
            #faces = frCC.detectMultiScale(skin) #TODO, check when to apply profile view and its effect on structural similarity
            print  "Found {0} faces!".format(len(faces))
            if not refFaceFound and len(faces)>0:
                if len(faces)>1:
                    # If multiple faces found in reference frame, go to next frame
                    print "checking for the next frame"
                    #continue
                # Save the First Image as Reference
                (x,y,w,h) = faces[0]

                #filePath = saveFace(aUId, faceCount, frame[y:y+h,x:x+w])
                #faceCount = faceCount+1
                #im_size, trainImage = getRefImage(filePath)
                #refFaceFound = 1 

            for (x,y,w,h) in faces:
                #faceImage = getFaceImage(skin, frame, im_size, x, y, w, h)
		gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                face_name = "Sample" + str(faceCount) + ".jpg"
		localizer = Flandmark()
		keypoints = localizer.locate(gray_image, y, x, h, w)
		print keypoints
		#print len(keypoints)
		if (keypoints) == None:
			continue	
		left_x = (keypoints[5][1] + keypoints[1][1])/2
        	left_y = (keypoints[5][0] + keypoints[1][0])/2
        	right_x = (keypoints[6][1] + keypoints[2][1])/2
        	right_y = (keypoints[6][0] + keypoints[2][0])/2
		cv.imwrite("Image.jpg", frame)
		img = Image.open("Image.jpg")
		
		fileDir = os.path.join(gOutputDir, aUId)
    		createDir(fileDir)
    		filePath = os.path.join(fileDir, face_name)
		print filePath
		cropfaces.CropFace(img, eye_left = (left_x, left_y), eye_right = (right_x, right_y), offset_pct =(0.3, 0.3), dest_sz = (200, 200)).save(filePath)
		# Computing the structural similarity measure for false positive rejection
                #cv.imshow("skindetected", frame)
		#cv.waitKey(100)
		#s = ssim(faceImage, trainImage)
                #if s > 0.00:
                #saveFace(aUId, faceCount, frame[y:y+h,x:x+w])
                faceCount = faceCount+1
        i = i + 1

    # Cleanup the video and close any open windows
    video.release()
    cv.destroyAllWindows()

def getImagesAndLabels(aPath, aUId):
    # Images will contains face images
    images = []
    # Labels will contains the label that is assigned to the image
    labels = []
    # Append all the absolute image paths in a list image_path
    imagePaths = [os.path.join(aPath, filePath) for filePath in os.listdir(aPath)]
    for imagePath in imagePaths:
        # Read the image and convert to grayscale
        imagePil = Image.open(imagePath).convert('L')
        # Convert the image format into numpy array
        image = np.array(imagePil, 'uint8')
        # append the images
        images.append(image)
        labels.append(int(aUId))
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
        return
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
    print aVideoPath
    video = cv.VideoCapture(aVideoPath)
    logging.debug('Loading the test video')
    frCC = cv.CascadeClassifier(gFrCascadePath)
    recognizers = loadFeatureVectors()

    # keep looping over the frames in the video
    i = 1
    gFaceCount = 1
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break
        if i%10 == 0:
            skin = getSkin(frame)
            logging.debug('skin detection was successfully completed')
            faces = frCC.detectMultiScale(skin, 1.4, 10)
            logging.debug('Face detection was successfully completed')
            #logging.debug('Found {0} faces', len(faces))
            recognizer = cv.createLBPHFaceRecognizer()
	    for (x,y,w,h) in faces:
                cv.rectangle(skin, (x,y),(x+w, y+h), (0,255,0), 2)
                Face = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                localizer = Flandmark()
                keypoints = localizer.locate(Face, y, x, h, w)
                # print keypoints
		if keypoints == None:
			continue
		left_x = (keypoints[5][1] + keypoints[1][1])/2
                left_y = (keypoints[5][0] + keypoints[1][0])/2
                right_x = (keypoints[6][1] + keypoints[2][1])/2
                right_y = (keypoints[6][0] + keypoints[2][0])/2
                
		cv.imwrite("Image.jpg", frame)
                img = Image.open("Image.jpg")
		cropfaces.CropFace(img, eye_left = (left_x, left_y), eye_right = (right_x, right_y), offset_pct =(0.3, 0.3), dest_sz = (200, 200)).save("Face.jpg")
		# Apply all the feature vectors on this face
		recognizer.load(os.path.join(gOutputDir, 'lbph.xml'))
                #for recognizer in recognizers:
                #print recognizer
		Res = imread("Face.jpg")
		Resimg = cv.cvtColor(Res, cv.COLOR_BGR2GRAY)
                nbr_predicted, conf = recognizer.predict(Resimg)
                if conf<100:
                    print nbr_predicted, conf
		    cv.imshow(gnamepair[nbr_predicted], frame[y:y+h, x:x+w])
                    cv.waitKey(100)
		    gres[nbr_predicted-1] = gres[nbr_predicted-1] + 1
	    	    gFaceCount = gFaceCount + 1
       		    print gFaceCount
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
       # saveFeatureVector(uid)
    else:
        print "Applying Data for Professor"
        recognizeFace(videoPath)
	max_value = max(gres)
        max_index = gres.index(max_value)
        Prof =  cv.imread(str(max_index+1)+".jpg")
        cv.imshow(gnamepair[max_index+1], Prof)
        cv.waitKey(500000)
        #print str(max_index+1) + "is the recognized professor"  
        print "Done"
