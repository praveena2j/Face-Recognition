# Common Utility functions for face detection library
import os
# Creates a directory if it doen't exists
def createDir(aDirPath):
    # If directory does not exist create it
    if not os.path.exists(aDirPath):
        os.makedirs(aDirPath)

# Deletes a file if it exists 
def deleteFile(aFile):
    try:
        os.remove(aFile)
    except OSError:
        pass

# Check if a file exists 
def existsFile(aFile):
    if (os.path.exists(aFile) and os.path.isfile(aFile)):
        return True
    else:
        return False
