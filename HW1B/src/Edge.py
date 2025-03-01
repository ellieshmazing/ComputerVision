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
        
#Function to stretch image to fit 8Bit range
#Input: Input image, max possible value, min possible value
#Output: Displayable image
def stretchToDisplayRange(img, minPos, maxPos):
    return ((img - minPos) * (255 / (maxPos - minPos)))

#Function to perform max LCS to display range
#Input: Input image
#Output: Displayable image
def displayRangeLCS(img):
    return (img - np.min(img)) * (255 / (np.max(img) - np.min(img)))

#Function to compute x-gradient for image
#Input: Single-channel image
#Output: x-gradient image (POSSIBLE MIN: = -1020 POSSIBLE MAX = 1020)
def xGrad(img):
    #Calculate image filtered with y-direction Sobel
    return cv.Sobel(img, cv.CV_64F, 1, 0, 1)

#Function to compute y-gradient for image
#Input: Single-channel image
#Output: y-gradient image (POSSIBLE MIN: = -1020 POSSIBLE MAX = 1020)
def yGrad(img):
    #Calculate image filtered with y-direction Sobel
    return cv.Sobel(img, cv.CV_64F, 0, 1, 1)

#Function to compute top-left tensor element
#Input: RGB image and sigma value for Gaussian filter
#Output: Top-left tensor element
def topLeftElem(img, sigma):
    #Compute first x-derivative of image for each color channel
    xGradB = xGrad(img[:,:,0])
    xGradG = xGrad(img[:,:,1])
    xGradR = xGrad(img[:,:,2])
    
    #Raise all array elements to power of two for each channel (POSSIBLE MIN: = 0 POSSIBLE MAX = 1040400)
    xGradSquareB = np.power(xGradB, 2)
    xGradSquareG = np.power(xGradG, 2)
    xGradSquareR = np.power(xGradR, 2)
    
    #Sum values
    xGradSum = np.add(xGradSquareB, np.add(xGradSquareG, xGradSquareR))
    
    #Sum each color channel to produce final tensor and return (POSSIBLE MIN: = 0 POSSIBLE MAX = 3121200)
    return cv.GaussianBlur(xGradSum, (0,0), sigma)

#Function to compute gradient magnitude
#Input: x- and y-gradients
#Output: Gradient magnitude image
def gradMag(xGrad, yGrad):
    return np.sqrt(np.pow(xGrad, 2) + np.pow(yGrad, 2))

#Function to compute bottom-right tensor element
#Input: RGB image and sigma value for Gaussian filter
#Output: Bottom-right tensor element
def bottomRightElem(img, sigma):
    #Compute first y-derivative of image for each color channel
    yGradB = yGrad(img[:,:,0])
    yGradG = yGrad(img[:,:,1])
    yGradR = yGrad(img[:,:,2])
    
    #Raise all array elements to power of two for each channel (POSSIBLE MIN: = 0 POSSIBLE MAX = 1040400)
    yGradSquareB = np.power(yGradB, 2)
    yGradSquareG = np.power(yGradG, 2)
    yGradSquareR = np.power(yGradR, 2)
    
    #Sum values
    yGradSum = np.add(yGradSquareB, np.add(yGradSquareG, yGradSquareR))
    
    #Sum each color channel to produce final tensor and return (POSSIBLE MIN: = 0 POSSIBLE MAX = 3121200)
    return cv.GaussianBlur(yGradSum, (0,0), sigma)

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
    
    #Multiply x- and y-derivatives for each channel (POSSIBLE MIN: = -1040400 POSSIBLE MAX = 1040400)
    xyMultB = np.multiply(xGradB, yGradB)
    xyMultG = np.multiply(xGradG, yGradG)
    xyMultR = np.multiply(xGradR, yGradR)
    
    #Sum values
    xyMultSum = np.add(xyMultB, np.add(xyMultG, xyMultR))
    
    #Sum each color channel to produce final tensor and return (POSSIBLE MIN: = -3121200 POSSIBLE MAX = 3121200)
    return cv.GaussianBlur(xyMultSum, (0,0), sigma)

#Function to apply 2D Color Structure Tensor
#Input: RGB image and sigma value for Gaussian filter + Output and filename array
#Output: Elements of tensor and their combined trace
def colorStructureTensor(img, sigma, imgOutputs, imgNames):
#Calculate top-left element - save display range and LCS versions
    tensorTLRaw = topLeftElem(img, sigma)
    
    tensorTL = stretchToDisplayRange(tensorTLRaw, 0, 3121200).astype(np.uint8)
    imgOutputs.append(tensorTL)
    imgNames.append("TensorTL")
    
    tensorTLLCS = displayRangeLCS(tensorTLRaw).astype(np.uint8)
    imgOutputs.append(tensorTLLCS)
    imgNames.append("TensorTLLCS")
    
    #Calculate bottom-right element and convert to display range
    tensorBRRaw = bottomRightElem(img, sigma)
    
    tensorBR = stretchToDisplayRange(tensorBRRaw, 0, 3121200).astype(np.uint8)
    imgOutputs.append(tensorBR)
    imgNames.append("TensorBR")
    
    tensorBRLCS = displayRangeLCS(tensorBRRaw).astype(np.uint8)
    imgOutputs.append(tensorBRLCS)
    imgNames.append("TensorBRLCS")
    
    #Calculate top-right/bottom-left tensor element
    tensorOtherRaw = otherElems(img, sigma)
    
    tensorOther = stretchToDisplayRange(tensorOtherRaw, -3121200, 3121200).astype(np.uint8)
    imgOutputs.append(tensorOther)
    imgNames.append("TensorOther")
    
    tensorOtherLCS = displayRangeLCS(tensorOtherRaw).astype(np.uint8)
    imgOutputs.append(tensorOtherLCS)
    imgNames.append("TensorOtherLCS")
    
    #Calculate tensor trace
    traceRaw = tensorTrace(tensorTLRaw, tensorBRRaw)
    
    trace = stretchToDisplayRange(traceRaw, 0, 6242400).astype(np.uint8)
    imgOutputs.append(trace)
    imgNames.append("TensorTrace")
    
    traceLCS = displayRangeLCS(traceRaw).astype(np.uint8)
    imgOutputs.append(traceLCS)
    imgNames.append("TensorTraceLCS")

    #Return output arrays with new members
    return imgOutputs, imgNames
    
#Function to calculate trace of tensor matrix
#Input: Top-left and bottom-right tensor elements
#Output: Trace (POSSIBLE MIN: = 0 POSSIBLE MAX = 6242400)
def tensorTrace(topLeft, bottomRight):
    return np.add(topLeft, bottomRight)


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
imgOutputs.append(stretchToDisplayRange(xGrad(img[:,:,0]), -1020, 1020).astype(np.uint8))
imgNames.append("xGradB")
imgOutputs.append(stretchToDisplayRange(xGrad(img[:,:,1]), -1020, 1020).astype(np.uint8))
imgNames.append("xGradG")
imgOutputs.append(stretchToDisplayRange(xGrad(img[:,:,2]), -1020, 1020).astype(np.uint8))
imgNames.append("xGradR")


#Compute and save y-axis image gradient for all color channels
imgOutputs.append(stretchToDisplayRange(yGrad(img[:,:,0]), -1020, 1020).astype(np.uint8))
imgNames.append("yGradB")
imgOutputs.append(stretchToDisplayRange(yGrad(img[:,:,1]), -1020, 1020).astype(np.uint8))
imgNames.append("yGradG")
imgOutputs.append(stretchToDisplayRange(yGrad(img[:,:,2]), -1020, 1020).astype(np.uint8))
imgNames.append("yGradR")


#Compute and save gradients for grayscale image
imgOutputs.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
imgNames.append("Gray")

grayXGrad = xGrad(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
imgOutputs.append(stretchToDisplayRange(grayXGrad, -1020, 1020).astype(np.uint8))
imgNames.append("GrayXGrad")

grayYGrad = yGrad(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
imgOutputs.append(stretchToDisplayRange(grayYGrad, -1020, 1020).astype(np.uint8))
imgNames.append("GrayYGrad")

grayGradMag = gradMag(grayXGrad, grayYGrad)
imgOutputs.append(stretchToDisplayRange(grayGradMag, 0, 1442.49783).astype(np.uint8))
imgNames.append("GrayMagGrad")

grayGradMagLCS = displayRangeLCS(grayGradMag).astype(np.uint8)
imgOutputs.append(grayGradMagLCS)
imgNames.append("GrayMagGradLCS")


#Compute and save 2D Color Structure Tensor Elements
imgOutputs, imgNames = colorStructureTensor(img, sigma, imgOutputs, imgNames)


#Save all output images
saveImages(outPath, imgOutputs, imgNames)