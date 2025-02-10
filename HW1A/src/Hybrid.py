'''Program to create hybrid images from two inputs'''
import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt

#Function to save output images
#Input: Output path, array of images, array of image names
#Output: None
def saveImages(outPath, imgOutputs, imgNames):
    #Iterate through imgOutputs and save images by corresponding title in imgNames
    for index, img in enumerate(imgOutputs):
        cv.imwrite(outPath + imgNames[index] + ".jpg", img)

#Function to generate color-coded histograms from input images
#Input: Output path, array of images, array of image names
#Output: None       
def generateHistograms(outPath, imgOutputs, imgNames):    
    #Create output directory if nonexistent
    if (not os.path.exists(outPath)):
        os.mkdir(outPath)
        
    #Iterate through imgOutputs, generate histograms for all three channels of each, and add to imgOutputs
    for index, img in enumerate(imgOutputs):
        #Clear plot
        plt.clf()
        
        #Generate and plot histogram for blue channel
        histB = cv.calcHist([img], [0], None, [256], [0,256])
        plt.plot(histB, color = 'b')
    
        #Generate and plot histogram for green channel
        histG = cv.calcHist([img], [1], None, [256], [0,256])
        plt.plot(histG, color = 'g')
        
        #Generate and plot histogram for red channel
        histR = cv.calcHist([img], [2], None, [256], [0,256])
        plt.plot(histR, color = 'r')

        #Save histogram to output folder
        plt.savefig(outPath + imgNames[index] + 'Hist.jpg')
        
#Function to generate smoothed image
#Input: Source image and sigma value for Gaussian filter
#Output: Smoothed image
def smoothImage(img, sigma):
    return cv.GaussianBlur(img, (0,0), sigma)

#Function to apply high-pass filter to image
#Input: Source image and sigma value for Gaussian filter
#Output: Filtered image
def highpassImage(img, sigma):
    return cv.subtract(img, cv.GaussianBlur(img, (0,0), sigma))

#Function to generate sharpened version of image
#Input: Source image and sigma value for Gaussian filter
#Output: Sharpened image
def sharpenImage(img, sigma):
    return cv.add(img, highpassImage(img, sigma))

#Function to generate hybrid images of two inputs
#Input: Source images and sigma value for Gaussian filter
#Output: Both possible hybrid images
def hybridImage(imgOne, imgTwo, sigma):
    hybridOneTwo = cv.add(smoothImage(imgOne, sigma), highpassImage(imgTwo, sigma))
    hybridTwoOne = cv.add(smoothImage(imgTwo, sigma), highpassImage(imgOne, sigma))
    
    return hybridOneTwo, hybridTwoOne
    


#Get paths for input images, output directory, and sigma value
srcDir = os.path.dirname(os.path.abspath(__file__))
inPathOne = str(srcDir + '\\' + sys.argv[1])
inPathTwo = str(srcDir + '\\' + sys.argv[2])
outPath = str(srcDir + '\\' + sys.argv[3])
sigma = float(sys.argv[4])


#Read in input images
imgOne = cv.imread(inPathOne)
imgTwo = cv.imread(inPathTwo)


#Create arrays to hold output images and their filenames
imgOutputs = []
imgNames = []


#Add input images to ouput array
imgOutputs.append(imgOne)
imgNames.append("InputOne")

imgOutputs.append(imgTwo)
imgNames.append("InputTwo")


#Smooth images based on given sigma and add to output array
imgSmoothOne = smoothImage(imgOne, sigma)
imgOutputs.append(imgSmoothOne)
imgNames.append("SmoothOne")

imgSmoothTwo = smoothImage(imgTwo, sigma)
imgOutputs.append(imgSmoothTwo)
imgNames.append("SmoothTwo")


#Apply high-pass filter to the images, and add to output array
imgHighOne = highpassImage(imgOne, sigma)
imgOutputs.append(imgHighOne)
imgNames.append("HighOne")

imgHighTwo = highpassImage(imgTwo, sigma)
imgOutputs.append(imgHighTwo)
imgNames.append("HighTwo")


#Generate sharpened images, and add to output array
imgSharpOne = sharpenImage(imgOne, sigma)
imgOutputs.append(imgSharpOne)
imgNames.append("SharpOne")

imgSharpTwo = sharpenImage(imgTwo, sigma)
imgOutputs.append(imgSharpTwo)
imgNames.append("SharpTwo")


#Generate both possible hybrid images, and add to output array
hybridOneTwo, hybridTwoOne = hybridImage(imgOne, imgTwo, sigma)
imgOutputs.append(hybridOneTwo)
imgNames.append("HybridOne")

imgOutputs.append(hybridTwoOne)
imgNames.append("HybridTwo")

#Generate histograms
generateHistograms(outPath, imgOutputs, imgNames)

#Save all outputs images
saveImages(outPath, imgOutputs, imgNames)