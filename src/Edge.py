'''Program to perform edge detection on RGB/BW images'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Function to save output images
#Input: Output path, array of images, array of image names
#Output: None
def saveImages(outPath, imgOutputs, imgNames):
    #Create output directory if nonexistent
    if (not os.path.exists(outPath)):
        os.mkdir(outPath)
        
    #Iterate through imgOutputs and save images by corresponding title in imgNames
    for index, img in enumerate(imgOutputs):
        cv.imwrite(outPath + imgNames[index] + ".jpg", img)
        cv.imwrite(outPath + imgNames[index] + "LCS.jpg", displayRangeLCS(img).astype(np.uint8))
        
#Function to stretch image to fit 8Bit range
#Input: Input image, max possible value, min possible value
#Output: Displayable image
def stretchToDisplayRange(img, minPos, maxPos):
    return ((img - minPos) * (255 / (maxPos - minPos))).astype(int)

#Function to perform max LCS to display range
#Input: Input image
#Output: Displayable image
def displayRangeLCS(img):
    return (img - np.min(img)) * (255 / (np.max(img) - np.min(img))).astype(int)
        
#Function to compute x-gradient for image
#Input: Single-channel image
#Output: x-gradient image
def xGrad(img):
    #Calculate image filtered with y-direction Sobel
    imgXGrad = cv.Sobel(img, cv.CV_16S, 1, 0, 3)

    #Return image stretched to displayable range
    return stretchToDisplayRange(imgXGrad, -1020, 1020)

#Function to compute y-gradient for image
#Input: Single-channel image
#Output: y-gradient image
def yGrad(img):
    #Calculate image filtered with y-direction Sobel
    imgYGrad = cv.Sobel(img, cv.CV_16S, 0, 1, 3)

    #Return image stretched to displayable range
    return stretchToDisplayRange(imgYGrad, -1020, 1020)

#Function to compute top-left tensor element
#Input: RGB image and sigma value for Gaussian filter
#Output: Top-left tensor element
def topLeftElem(img, sigma):
    #Compute first x-derivative of image for each color channel
    xGradB = xGrad(img[:,:,0])
    xGradG = xGrad(img[:,:,1])
    xGradR = xGrad(img[:,:,2])
    
    #Raise all array elements to power of two for each channel, then fit to 8Bit range
    xGradSquareB = stretchToDisplayRange(np.power(xGradB, 2), 0, 65025)
    xGradSquareG = stretchToDisplayRange(np.power(xGradG, 2), 0, 65025)
    xGradSquareR = stretchToDisplayRange(np.power(xGradR, 2), 0, 65025)
    
    #Apply Gaussian filter to each squared channel
    smoothB = cv.GaussianBlur(xGradSquareB.astype(np.uint8), (0,0), sigma)
    smoothG = cv.GaussianBlur(xGradSquareG.astype(np.uint8), (0,0), sigma)
    smoothR = cv.GaussianBlur(xGradSquareR.astype(np.uint8), (0,0), sigma)
    
    #Sum each color channel to produce final tensor and return
    return cv.add(smoothB, cv.add(smoothG, smoothR))

#Function to compute bottom-right tensor element
#Input: RGB image and sigma value for Gaussian filter
#Output: Bottom-right tensor element
def bottomRightElem(img, sigma):
    #Compute first y-derivative of image for each color channel
    yGradB = yGrad(img[:,:,0])
    yGradG = yGrad(img[:,:,1])
    yGradR = yGrad(img[:,:,2])
    
    #Raise all array elements to power of two for each channel, then fit to 8Bit range
    yGradSquareB = stretchToDisplayRange(np.power(yGradB, 2), 0, 65025)
    yGradSquareG = stretchToDisplayRange(np.power(yGradG, 2), 0, 65025)
    yGradSquareR = stretchToDisplayRange(np.power(yGradR, 2), 0, 65025)
    
    #Apply Gaussian filter to each squared channel
    smoothB = cv.GaussianBlur(yGradSquareB.astype(np.uint8), (0,0), sigma)
    smoothG = cv.GaussianBlur(yGradSquareG.astype(np.uint8), (0,0), sigma)
    smoothR = cv.GaussianBlur(yGradSquareR.astype(np.uint8), (0,0), sigma)
    
    #Sum each color channel to produce final tensor and return
    return cv.add(smoothB, cv.add(smoothG, smoothR))

#Function to compute top-left/bottom-right tensor elements
#Input: RGB image and sigma value for Gaussian filter
#Output: Top-left/botton-right tensor element
def otherElems(img, sigma):
    #Compute first x-derivative of image for each color channel
    xGradB = xGrad(img[:,:,0])
    xGradG = xGrad(img[:,:,1])
    xGradR = xGrad(img[:,:,2])
    
    #Compute first y-derivative of image for each color channel
    yGradB = yGrad(img[:,:,0])
    yGradG = yGrad(img[:,:,1])
    yGradR = yGrad(img[:,:,2])
    
    #Multiply x- and y-derivatives for each channel, then fit to 8Bit range
    xyMultB = stretchToDisplayRange(np.multiply(xGradB, yGradB), 0, 65025)
    xyMultG = stretchToDisplayRange(np.multiply(xGradG, yGradG), 0, 65025)
    xyMultR = stretchToDisplayRange(np.multiply(xGradR, yGradR), 0, 65025)
    
    #Apply Gaussian filter to each squared channel
    smoothB = cv.GaussianBlur(xyMultB.astype(np.uint8), (0,0), sigma)
    smoothG = cv.GaussianBlur(xyMultG.astype(np.uint8), (0,0), sigma)
    smoothR = cv.GaussianBlur(xyMultR.astype(np.uint8), (0,0), sigma)
    
    #Sum each color channel to produce final tensor and return
    return cv.add(smoothB, cv.add(smoothG, smoothR))

#Function to apply 2D Color Structure Tensor
#Input: RGB image and sigma value for Gaussian filter + Output and filename array
#Output: Elements of tensor and their combined trace
def colorStructureTensor(img, sigma, imgOutputs, imgNames):
    #Calculate top-left element and convert to display range
    tensorTL = stretchToDisplayRange(topLeftElem(img, sigma), 0 , 765).astype(np.uint8)
    imgOutputs.append(tensorTL)
    imgNames.append("TensorTL")
    
    #Calculate bottom-right element and convert to display range
    tensorBR = stretchToDisplayRange(bottomRightElem(img, sigma), 0 , 765).astype(np.uint8)
    imgOutputs.append(tensorBR)
    imgNames.append("TensorBR")
    
    #Calculate top-right/bottom-left tensor element
    tensorOther = stretchToDisplayRange(otherElems(img, sigma), 0 , 765).astype(np.uint8)
    imgOutputs.append(tensorOther)
    imgNames.append("TensorOther")
    
    #Calculate tensor trace
    trace = stretchToDisplayRange(tensorTrace(tensorTL, tensorBR), 0, 510).astype(np.uint8)
    imgOutputs.append(trace)
    imgNames.append("TensorTrace")

    #Return output arrays with new members
    return imgOutputs, imgNames
    
#Function to calculate trace of tensor matrix
#Input: Top-left and bottom-right tensor elements
#Output: Trace, stretched to visible range
def tensorTrace(topLeft, bottomRight):
    return cv.add(topLeft, bottomRight)


#Get paths for input images, output directory, and small/large-scale sigma values
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])
sigma = float(sys.argv[3])


#Read in input image
img = cv.imread(inPath)


#Create arrays to hold output images and their filenames, and add input
imgOutputs = []
imgNames = []

imgOutputs.append(img)
imgNames.append("Input")

#Compute and save x-axis image gradient for all color channels
imgOutputs.append(xGrad(img[:,:,0]))
imgNames.append("xGradB")
imgOutputs.append(xGrad(img[:,:,1]))
imgNames.append("xGradG")
imgOutputs.append(xGrad(img[:,:,2]))
imgNames.append("xGradR")


#Compute and save y-axis image gradient for all color channels
imgOutputs.append(yGrad(img[:,:,0]))
imgNames.append("yGradB")
imgOutputs.append(yGrad(img[:,:,1]))
imgNames.append("yGradG")
imgOutputs.append(yGrad(img[:,:,2]))
imgNames.append("yGradR")


#Compute and save 2D Color Structure Tensor Elements
imgOutputs, imgNames = colorStructureTensor(img, sigma, imgOutputs, imgNames)


#Save all output images
saveImages(outPath, imgOutputs, imgNames)