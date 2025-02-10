import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Display image in window
def displayImage(img):
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#Save image to given path
def saveImage(img, path):
    srcDir = os.path.dirname(os.path.abspath(__file__))
    outPath = str(srcDir + '\\' + path)

    cv.imwrite(outPath, img)

#Convert RGB image to grayscale
def rgb2gray(img):
    #Extract necessary size information
    imgHeight, imgWidth = img.shape[:2]

    #Create numpy array for new image
    imgGray = np.empty([imgHeight, imgWidth], dtype = np.uint8)

    #Iterate through new image and calculate value for each pixel
    for y in range(imgHeight):
        for x in range(imgWidth):
            imgGray[y][x] = int(.1140 * img[y][x][0] + .5871 * img[y][x][1] + .2989 * img[y][x][2])

    return imgGray

#Convert RGB image to gray using single channel
def rgb2graySingleChannel(img, channel):
    #Extract necessary size information
    imgHeight, imgWidth = img.shape[:2]

    #Iterate through new image and calculate value for each pixel
    for y in range(imgHeight):
        for x in range(imgWidth):
            img[y][x][0] = img[y][x][channel]
            img[y][x][1] = img[y][x][channel]
            img[y][x][2] = img[y][x][channel]

    return img

#Process image using binary threshold set at given threshold for given channel
def rgb2binary(img, channel, threshold):
    #Extract necessary size information
    imgHeight, imgWidth = img.shape[:2]

    #Iterate through image and modify each pixel by blue threshold
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (img[y][x][channel] > threshold):
                img[y][x][0] = 255
                img[y][x][1] = 255
                img[y][x][2] = 255
            else:
                img[y][x][0] = 0
                img[y][x][1] = 0
                img[y][x][2] = 0

    return img

#Process grayscale image for given image with given threshold
def gray2binary(img, threshold):
    #Extract size information
    imgHeight, imgWidth = img.shape[:2]

    #Iterate through each pixel and modify according to threshold
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (img[y][x] > threshold):
                img[y][x] = 255
            else:
                img[y][x] = 0

    return img

#Generate histogram of single channel of image
def histogram (img, nBins, channel):
    #Extract size attributes
    imgHeight, imgWidth = img.shape[:2]

    #Create numpy array to hold histogram data
    resultHist = np.zeros(nBins)

    #Determine bin size in intensity range
    binWidth = 256 / nBins

    for y in range(imgHeight):
        for x in range(imgWidth):
            pixelBin = int(img[y][x][channel] / binWidth)
            resultHist[pixelBin] = resultHist[pixelBin] + 1

    return resultHist

#Generate histogram of masked section of image
def maskedHistogram (img, mask, nBins, channel):
    #Extract size attributes
    imgHeight, imgWidth = img.shape[:2]

    #Create numpy array to hold histogram data
    resultHist = np.zeros(nBins)

    #Determine bin size in intensity range
    binWidth = 256 / nBins

    for y in range(imgHeight):
        for x in range(imgWidth):
            if (mask[y][x][0] != 0):
                pixelBin = int(img[y][x][channel] / binWidth)
                resultHist[pixelBin] = resultHist[pixelBin] + 1

    return resultHist

#Get mean of histogram values
def histMean(imgHist, pixelCount, nBins):
    mean = 0
    for x in range(nBins):
        mean += x * (imgHist[x] / pixelCount)
        
    return mean

#Display histogram as bar graph
def displayHist(data, nBins):
    x = np.arange(nBins)
    fig, ax = plt.subplots()
    ax.bar(x, data, width=1, edgecolor="white", linewidth=0.7)

    plt.show()

def minMaxLCS(img, channel):
    #Generate histogram of image intensity values
    imgHist = histogram(img, 256, channel)
    
    #Get minimum and maximum intensity values
    minVal = 256
    maxVal = 0 
    for x in range(0, 256):
        if (imgHist[x] != 0):
            if (x < minVal):
                minVal = x
            if (x > maxVal):
                maxVal = x
                
    #Calculate new value for each intensity level
    diff = maxVal - minVal
    newVals = np.arange(maxVal + 1)
    for x in range(0, maxVal + 1):
        newVals[x] = int(255 * ((x - minVal) / diff))
        
    #Change each pixel to new intensity value
    imgHeight, imgWidth = img.shape[:2]

    for y in range(imgHeight):
        for x in range(imgWidth):
            img[y][x] = newVals[img[y][x]]
            
    return img

def dropLCS(img, channel, dropPercent):
    #Get image size and calculate pixels to drop
    imgHeight, imgWidth = img.shape[:2]
    imgPixelCount = imgHeight * imgWidth
    dropCount = int(imgPixelCount * dropPercent / 200) #200 since pixels will be removed from both sides

    #Generate histogram of image intensity values
    imgHist = histogram(img, 256, channel)

    #Get minimum and maximum intensity values
    minVal = 0
    lowCount = 0
    for x in range(0, 256):
        if (imgHist[x] != 0):
            lowCount += imgHist[x]
            if (lowCount > dropCount):
                minVal = x
                break

    maxVal = 0
    highCount = 0
    for x in reversed(range(0, 256)):
        if (imgHist[x] != 0):
            highCount += imgHist[x]
            if (highCount > dropCount):
                maxVal = x
                break
            
    #Calculate new value for each intensity level
    diff = maxVal - minVal
    newVals = np.arange(255)
    for x in range(0, 255):
        newVals[x] = int(255 * ((x - minVal) / diff))
        if (newVals[x] < 0):
            newVals[x] = 0
        if (newVals[x] > 255):
            newVals[x] = 255

    #Change each pixel to new intensity value
    for y in range(imgHeight):
        for x in range(imgWidth):
            img[y][x] = newVals[img[y][x]]
            
    return img

def dropTopLCS(img, channel, dropPercent):
    #Get image size and calculate pixels to drop
    imgHeight, imgWidth = img.shape[:2]
    imgPixelCount = imgHeight * imgWidth
    dropCount = int(imgPixelCount * dropPercent / 100) #200 since pixels will be removed from both sides

    #Generate histogram of image intensity values
    imgHist = histogram(img, 256, channel)

    #Get minimum and maximum intensity values
    minVal = 0
    for x in range(0, 256):
        if (imgHist[x] != 0):
            minVal = x
            break

    maxVal = 0
    highCount = 0
    for x in reversed(range(0, 256)):
        if (imgHist[x] != 0):
            highCount += imgHist[x]
            if (highCount > dropCount):
                maxVal = x
                break
            
    #Calculate new value for each intensity level
    diff = maxVal - minVal
    newVals = np.arange(256)
    for x in range(0, 256):
        newVals[x] = int(255 * ((x - minVal) / diff))
        if (newVals[x] < 0):
            newVals[x] = 0
        if (newVals[x] > 255):
            newVals[x] = 255

    #Change each pixel to new intensity value
    for y in range(imgHeight):
        for x in range(imgWidth):
            val = img[y][x][channel]
            for newChannel in range(3):
                img[y][x][newChannel] = newVals[val]
            
    return img

def dropBotLCS(img, channel, dropPercent):
    #Get image size and calculate pixels to drop
    imgHeight, imgWidth = img.shape[:2]
    imgPixelCount = imgHeight * imgWidth
    dropCount = int(imgPixelCount * dropPercent / 100) #200 since pixels will be removed from both sides

    #Generate histogram of image intensity values
    imgHist = histogram(img, 256, channel)

    #Get minimum and maximum intensity values
    maxVal = 0
    for x in reversed(range(0, 256)):
        if (imgHist[x] != 0):
            maxVal = x
            break
        
    minVal = 0
    lowCount = 0
    for x in range(0, 256):
        if (imgHist[x] != 0):
            lowCount += imgHist[x]
            if (lowCount > dropCount):
                minVal = x
                break
            
    #Calculate new value for each intensity level
    diff = maxVal - minVal
    newVals = np.arange(256)
    for x in range(0, 256):
        newVals[x] = int(255 * ((x - minVal) / diff))
        if (newVals[x] < 0):
            newVals[x] = 0
        if (newVals[x] > 255):
            newVals[x] = 255

    #Change each pixel to new intensity value
    for y in range(imgHeight):
        for x in range(imgWidth):
            val = img[y][x][channel]
            for newChannel in range(3):
                img[y][x][newChannel] = newVals[val]
            
    return img

#Wrapper function for recursive Otsu
def initOtsu(img, channel):
    #Get pixel count of image
    imgHeight, imgWidth = img.shape[:2]
    pixelCount = imgHeight * imgWidth
    
    #Generate histogram of image
    imgHist = histogram(img, 256, channel)
    
    #Calculate image mean
    mean = histMean(imgHist, pixelCount, 256)
    
    #Calculate base values for qOne and variation squared
    qOne = imgHist[0] / pixelCount
    varSquare = qOne * (1 - qOne) * pow(0 - 1, 2)
    
    #Call recursive function
    return recursiveOtsu(imgHist, pixelCount, mean, 1, qOne, 0, varSquare)
    
#Recursive Otsu function
def recursiveOtsu (imgHist, pixelCount, mean, thresh, qOne, uOne, varSquare):
    #Calculate next values for pOne and qOne
    pOnePlus = imgHist[thresh] / pixelCount
    qOnePlus = qOne + pOnePlus
    
    #Calculate the left class mean (with default to 0 if qOnePlus is 0 to avoid divide by zero error)
    if (qOnePlus > 0):
        uOnePlus = ((qOne * uOne) + (thresh * pOnePlus)) / qOnePlus
    else:
        uOnePlus = 0
    
    #Calculate the right class mean
    uTwoPlus = (mean - (qOnePlus * uOnePlus)) / (1 - qOnePlus)
    
    #Calculate the between-class variance with current threshold
    varSquarePlus = qOnePlus * (1 - qOnePlus) * pow(uOnePlus - uTwoPlus, 2)
        
    #If variance decreases, return previous value as it was maximum
    #This is due to the usage of binarization, as the variance will always be an inverted polynomial. With more classes,
    #ending execution early would not be possible
    if (varSquarePlus < varSquare):
        return thresh - 1
    
    #End execution if threshold is maximum value
    if (thresh == 255):
        return thresh
        
    #Execute for next threshold value
    return recursiveOtsu(imgHist, pixelCount, mean, thresh + 1, qOnePlus, uOnePlus, varSquarePlus)
    
#Determines Mean Square Error for two images
def meanSquareError(imgOne, imgTwo):
    #Extract size information and ensure images are compatible
    imgOneHeight, imgOneWidth = imgOne.shape[:2]
    imgTwoHeight, imgTwoWidth = imgTwo.shape[:2]
    if (imgOneHeight != imgTwoHeight or imgOneWidth != imgTwoWidth):
        print("These photos are not compatible")
        return

    #Declare variable to hold error
    error = 0.0
    
    #Calculate mean square error for every pixel and sum together
    for y in range(imgOneHeight):
        for x in range(imgOneWidth):
            diff = np.subtract(imgOne[y][x][0], imgTwo[y][x][0])
            
            if (diff != 0):            
                error += diff ** 2
            
    #Calculate and return average mean square error for the image
    error /= imgOneHeight * imgOneWidth
    return error