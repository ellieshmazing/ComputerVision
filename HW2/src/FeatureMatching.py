'''Program to perform edge detection on RGB/BW images'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from Edge import saveImages, topLeftElem, bottomRightElem, otherElems, tensorTrace, displayRangeLCS

#Function to calculate determinant of tensor matrix
#Input: Auto-correlation matrix
#Output: Determinant
def tensorDeterminant(matrix):
    return np.subtract(np.multiply(matrix[0], matrix[3]), np.multiply(matrix[1], matrix[2])) 

#Function to construct autocorrelation matrix from image
#Input: Source image and sigma value for Gaussian smoothing
#Output: Array of images [top-left, top-right, bottom-left, bottom-right]
def genAutoCorrelationMatrix(img, sigma):
    #Initiate output array
    matrix = []
    
    #Add each matrix element in order
    matrix.append(topLeftElem(img, sigma))
    matrix.append(otherElems(img, sigma))
    matrix.append(otherElems(img, sigma))
    matrix.append(bottomRightElem(img, sigma))
    
    #Return autocorrelation matrix
    return matrix

#Function to detect corners using Harris-Stephens Detector
#Input: Auto-correlation matrix of input image
#Output: Array of R values for whole image
def computeFeatureResponse(matrix):
    #Set value of empirical constant [0.04, 0.06]
    alpha = 0.05
    
    return np.subtract(tensorDeterminant(matrix), np.multiply(alpha, np.power(tensorTrace(matrix[0], matrix[3]), 2)))

#Function to return list of coordinates of most locally unique feature points
#Input: Array of R-values for an image and npoints number of feature points to return
#Output: List of feature point coordinates and binary mask showing location
#TO-DO: IMPLEMENT ANMS
def featureMeasure2Points(R, npoints):
    #Initialize list to hold feature coordinates
    featCoor = []
    
    #Initialize threshold based on global maximum
    threshVal = np.max(R) * .01
    
    #Calculate local maxima
    featCoor = peak_local_max(R, threshold_abs = threshVal)
    print(len(featCoor))
        
    #Connect feature coordinates to R-value
    featCoorVal = []
    for x in range(len(featCoor)):
        coorVal = []
        coorVal.append(featCoor[x])
        coorVal.append(R[featCoor[x][0]][featCoor[x][1]])
        featCoorVal.append(coorVal)
        
    #Sort feature coordinates by R-value, descending
    featCoorSorted = sorted(featCoorVal, key = lambda x: x[1], reverse = True)
    print(len(featCoorSorted))
    
    #Declare array to hold properly formatted feature coordinates and image of point mask
    featCoorFinal = []
    featMask = np.zeros(R.shape[:2], dtype=np.uint8)
    
    #Iterate through list of feature coordinates and add points to output array and mask
    for x in range(npoints):
        featCoorFinal.append(featCoorSorted[x][0])
        featMask[featCoorSorted[x][0][0]][featCoorSorted[x][0][1]] = 255
    
    #Return npoint largest maxima
    return featCoorFinal, featMask
        

#Get paths for input images, output directory, and small/large-scale sigma values
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])
sigma = float(sys.argv[3])
npoints = int(sys.argv[4])


#Read in input image
img = cv.imread(inPath)


#Create arrays to hold output images and their filenames, and add input
imgOutputs = []
imgNames = []

imgOutputs.append(img)
imgNames.append("Input")


#Generate auto-correlation matrix
matrix = genAutoCorrelationMatrix(img, sigma)


#Calculate feature response and save to output
rVals = computeFeatureResponse(matrix)

imgOutputs.append(rVals)
imgNames.append("Edges")
imgOutputs.append(displayRangeLCS(rVals))
imgNames.append("EdgesLCS")


#Calculate local maxima and save mask to output
featCoor, featCoorMask = featureMeasure2Points(rVals, npoints)

imgOutputs.append(featCoorMask)
imgNames.append("FeaturePointMask")

#Save all output images
saveImages(outPath, imgOutputs, imgNames)