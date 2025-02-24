'''Program to perform edge detection on RGB/BW images'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from Edge import saveImages, topLeftElem, bottomRightElem, otherElems, tensorTrace, displayRangeLCS

#Function to project feature points over image
#Input: Image and feature point mask
#output: Image with feature points projected
def projectFeaturePoints(img, featMask, rad):
    #Extract image shape
    imgHeight, imgWidth = img.shape[:2]
    
    #Copy image to allow alteration
    imgAlter = img.copy()

    #Iterate through image and add feature points
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (featMask[y][x] == 255):
                for yVar in range(rad):
                    for xVar in range(rad):
                        imgAlter[y+yVar][x+xVar][0] = 0
                        imgAlter[y+yVar][x+xVar][1] = 0
                        imgAlter[y+yVar][x+xVar][2] = 255
                        
                        imgAlter[y-yVar][x+xVar][0] = 0
                        imgAlter[y-yVar][x+xVar][1] = 0
                        imgAlter[y-yVar][x+xVar][2] = 255
                        
                        imgAlter[y+yVar][x-xVar][0] = 0
                        imgAlter[y+yVar][x-xVar][1] = 0
                        imgAlter[y+yVar][x-xVar][2] = 255
                        
                        imgAlter[y-yVar][x-xVar][0] = 0
                        imgAlter[y-yVar][x-xVar][1] = 0
                        imgAlter[y-yVar][x-xVar][2] = 255
              
    #Return modified image
    return imgAlter

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
        
    #Connect feature coordinates to R-value
    featCoorVal = []
    for x in range(len(featCoor)):
        coorVal = []
        coorVal.append(featCoor[x])
        coorVal.append(R[featCoor[x][0]][featCoor[x][1]])
        featCoorVal.append(coorVal)
        
    #Sort feature coordinates by R-value, descending
    featCoorSorted = sorted(featCoorVal, key = lambda x: x[1], reverse = True)
    
    #Declare array to hold properly formatted feature coordinates and image of point mask
    featCoorFinal = []
    featMask = np.zeros(R.shape[:2], dtype=np.uint8)
    
    #Iterate through list of feature coordinates and add points to output array and mask
    for x in range(npoints):
        featCoorFinal.append(featCoorSorted[x][0])
        featMask[featCoorSorted[x][0][0]][featCoorSorted[x][0][1]] = 255
    
    #Return npoint largest maxima
    return featCoorFinal, featMask

#Function to generate feature descriptors from list of feature points
#Input: Original image and list of feature coordinates
#Output: List of feature descriptors, indexed identically to the input feature coordinates
#TO-DO: IMPLEMENT BETTER DESCRIPTOR
def generateFeatureDescriptors(img, featCoor, rad):
    #Determine npoints from featCoor length
    npoints = len(featCoor)
    
    #Initialize array to hold descriptor list
    Dlist = []
    
    #Extract image size
    imgHeight, imgWidth = img.shape[:2]
    
    #Iterate through featCoor list and extract descriptor for each
    for i in range(npoints):
        #Declare array to hold 5x5 RGB image of descriptor
        descriptor = np.zeros((1 + rad * 2, 1 + rad * 2,3), dtype=np.uint8)
        
        #Translate pixels around feature coordinate from image to descriptor
        for y in range(rad + 1):
            for x in range(rad + 1):
                if (featCoor[i][0] + y < imgHeight and featCoor[i][1] + x < imgWidth):
                    descriptor[rad + y][rad + x] = img[featCoor[i][0] + y][featCoor[i][1] + x]
                else:
                    descriptor[rad + y][rad + x] = 0
                    
                if (featCoor[i][0] + y < imgHeight and featCoor[i][1] - x > -1):
                    descriptor[rad + y][rad - x] = img[featCoor[i][0] + y][featCoor[i][1] - x]
                else:
                    descriptor[rad + y][rad - x] = 0
                    
                if (featCoor[i][0] - y > -1 and featCoor[i][1] + x < imgWidth):
                    descriptor[rad - y][rad + x] = img[featCoor[i][0] - y][featCoor[i][1] + x]
                else:
                    descriptor[rad - y][rad + x] = 0
                    
                if (featCoor[i][0] - y > -1 and featCoor[i][1] - x > -1):
                    descriptor[rad - y][rad - x] = img[featCoor[i][0] - y][featCoor[i][1] - x]
                else:
                    descriptor[rad - y][rad - x] = 0
                
        #Add descriptor to Dlist
        Dlist.append(descriptor)
        
    #Return list of descriptors
    return Dlist

#Function to calculate L1 Norm distance between two descriptors
#Input: Two descriptors
#Output: L1 Norm
def l1Norm(desc1, desc2):
    #Initialize variable to hold sum of differences between pixels
    distSum = 0
    
    #Extract descriptor size
    imgHeight, imgWidth = desc1.shape[:2]
    
    #Iterate through descriptors and calculate individual pixel distance, than add to running sum
    for y in range(imgHeight):
        for x in range(imgWidth):
            distSum += np.abs(desc1[y][x] - desc2[y][x])
            
    #Return L1 Norm
    return distSum

#Function to calculate descriptor distances between two image descriptor lists
#Input: Dlist of image 1 and Dlist of image 2
#Output: Matrix, where Dist(i,j) is the distance between the ith descriptor of image 1 and the jth descriptor of image 2
#TO-DO: TAILOR TO NEW DESCRIPTOR GENERATOR
def computeDescriptorDistances(Dlist1, Dlist2):
    #Initialize matrix to hold distances
    Dist = []
    
    #Iterate through Dlists and calculate distances
    for i in range(len(Dlist1)):
        #Initialize list to hold distances from Dlist1[i]
        iDist = []
        
        for j in range(len(Dlist2)):
            #Append distance from Dlist2[j]
            iDist.append(l1Norm(Dlist1[i], Dlist2[j]))
            
        #Append list of distances to final matrix
        Dist.append(iDist)
        
    #Return matrix of distances
    return Dist    

#Function to determine nearest match of every descriptor across images
#Input: Distance matrix and threshold value
#Output: List of matches
def Distance2Matches_NearestMatch(Dist, Th2):
    #Initialize array to hold matches
    matchList = []
    
    #Iterate through distance matrix and determine match for each feature descriptor
    for i in range(len(Dist)):
        #Ensure minimum match is within threshold
        if (np.min(Dist[i]) < Th2):
            #Declare array to hold individual match
            iMatch = []
            iMatch.append(i)
        
            #Append index of minimum match
            iMatch.append(int(np.argmin(Dist[i])))
        
            #Append match for i to output list
            matchList.append(iMatch)
        
    #Return list of matches
    return matchList

#Get paths for input images, output directory, and small/large-scale sigma values
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath1 = str(srcDir + '\\' + sys.argv[1])
inPath2 = str(srcDir + '\\' + sys.argv[2])
outPath = str(srcDir + '\\' + sys.argv[3])
sigma = float(sys.argv[4])
npoints = int(sys.argv[5])


#Read in input image
img1 = cv.imread(inPath1)
img2 = cv.imread(inPath2)


#Create arrays to hold output images and their filenames, and add input
imgOutputs = []
imgNames = []

imgOutputs.append(img1)
imgNames.append("Input1")
imgOutputs.append(img2)
imgNames.append("Input2")


#Generate auto-correlation matrices
matrix1 = genAutoCorrelationMatrix(img1, sigma)
matrix2 = genAutoCorrelationMatrix(img2, sigma)


#Calculate feature response and save to output
rVals1 = computeFeatureResponse(matrix1)
rVals2 = computeFeatureResponse(matrix2)

imgOutputs.append(rVals1)
imgNames.append("Edges1")
imgOutputs.append(displayRangeLCS(rVals1))
imgNames.append("EdgesLCS1")

imgOutputs.append(rVals2)
imgNames.append("Edges2")
imgOutputs.append(displayRangeLCS(rVals2))
imgNames.append("EdgesLCS2")


#Calculate local maxima and save mask to output
featCoor1, featCoorMask1 = featureMeasure2Points(rVals1, npoints)
featCoor2, featCoorMask2 = featureMeasure2Points(rVals2, npoints)

imgOutputs.append(featCoorMask1)
imgNames.append("FeaturePointMask1")
imgOutputs.append(projectFeaturePoints(img1, featCoorMask1, 2))
imgNames.append("FeaturePoints1")

imgOutputs.append(featCoorMask2)
imgNames.append("FeaturePointMask2")
imgOutputs.append(projectFeaturePoints(img2, featCoorMask2, 2))
imgNames.append("FeaturePoints2")


#Generate list of descriptors, then matrix of distances
Dlist1 = generateFeatureDescriptors(img1, featCoor1, 10)
Dlist2 = generateFeatureDescriptors(img2, featCoor2, 10)
Dist = computeDescriptorDistances(Dlist1, Dlist2)


#Perform matching between descriptor lists
matchList2 = Distance2Matches_NearestMatch(Dist, 5)
print(len(matchList2))
print(matchList2[0])


#Save all output images
saveImages(outPath, imgOutputs, imgNames)