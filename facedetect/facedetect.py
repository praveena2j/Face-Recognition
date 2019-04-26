# Common image processing APIs for face detection
import cv2, numpy, os, sys, utils
# Define the upper and lower boundaries
# of the HSV pixel intensities to be considered 'skin'
gLower = numpy.array([0, 48, 0], dtype = "uint8")
gUpper = numpy.array([100, 255, 255], dtype = "uint8")

# Save a detected face in a file
# Input Params:
# @aUId: Unique ID of the person whoes face is detected
# @aFaceCount: count associated with the face
# @aOutputDir: Directory where the file needs to be saved
# @aFace: CV2 Image Paramters
# Returns: filePath of saved image 
def saveFace(aUId, aFaceCount, aOutputDir, aFace):
    fileName = str(aUId) + "." + str(aFaceCount) + ".png"
    fileDir = os.path.join(aOutputDir, aUId)
    utils.createDir(fileDir)
    filePath = os.path.join(fileDir, fileName)
    cv2.imwrite(filePath, aFace)
    return filePath

# Extracts a face from a video frame
# Input Params:
# @aSkin: TODO
# @aFrame: Video Frame
# @aRefSize: TODO
# @aX: TODO
# @aY: TODO
# @aW: TODO
# @aH: TODO
# Returns: faceImage 
def getFaceImage(aSkin, aFrame, aRefSize, aX, aY, aW, aH):
    cv2.rectangle(aSkin, (aX,aY),(aX+aW, aY+aH), (0,255,0), 2)
    tmpImage = imresize((aFrame[aY:aY+aH,aX:aX+aW]),(aRefSize[0], aRefSize[1]))
    faceImage = cv2.cvtColor(tmpImage, cv2.COLOR_BGR2GRAY)
    return faceImage

# Get portion of skin from a video frame
# Input Params:
# @aFrame: Video Frame
# Returns: Detected skin
def getSkin(aFrame):
    converted = cv2.cvtColor(aFrame, cv2.COLOR_BGR2HSV)
    # Determine the HSV pixel intensities that fall into the speicifed upper and lower boundaries
    skinMask = cv2.inRange(converted, gLower, gUpper)
    # Apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 6)
    # Blur the mask to help remove noise, then apply the mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(aFrame, aFrame, mask = skinMask)
    return skin

