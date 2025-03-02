'''Program to perform edge detection on RGB/BW images'''
import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from Edge import saveImages, topLeftElem, bottomRightElem, otherElems, xGrad, yGrad, tensorTrace, gradMag, displayRangeLCS


#Function to compute gradient orientation for an image
#Input: x- and y-gradients for image
#Output: Image-like array with gradient orientation values for each point
def gradOrientation(xGrad, yGrad):
    #Calculate arctan of gradients, default to vertical when xGradient is 0
    tempOr = np.arctan(np.divide(yGrad, xGrad, out = np.full_like(xGrad, np.pi / 2), where = xGrad != 0))
    
    #Extract image size attributes
    imgHeight, imgWidth = xGrad.shape[:2]
    
    #Iterate through orientations and adjust direction
    for y in range(imgHeight):
        for x in range(imgWidth):
            #Due to avoiding divide by 0, all gradients where xGrad = 0 are set to pi/2
            #Adjust these to negative if necessary
            if (tempOr[y][x] == np.pi / 2 and yGrad[y][x] < 0):
                tempOr[y][x] = -np.pi / 2
                
            #np.arctan returns within [-pi/2, pi/2], so adjust direction to other half of unit circle if xGrad is negative
            if (xGrad[y][x] < 0):
                tempOr[y][x] += np.pi
                
            #Add pi/2 to ensure positive values to simplify later calculations
            tempOr[y][x] += np.pi / 2
                
    #Return adjusted gradient calculations
    return tempOr

#Helper function to handle generating gradient magnitude and orientation for RGB image
#Input: RGB image
#Output: Gradient magnitude and orientation
def gradMagOrHelper(img):
    gradMagnitude = gradMag(np.add(xGrad(img[:,:,0]), np.add(xGrad(img[:,:,1]), xGrad(img[:,:,2]))), np.add(yGrad(img[:,:,0]), np.add(yGrad(img[:,:,1]), yGrad(img[:,:,2]))))
    gradOr = gradOrientation(np.add(xGrad(img[:,:,0]), np.add(xGrad(img[:,:,1]), xGrad(img[:,:,2]))), np.add(yGrad(img[:,:,0]), np.add(yGrad(img[:,:,1]), yGrad(img[:,:,2]))))
    return gradMagnitude, gradOr

#Function to project feature points over image
#Input: Image and feature point mask
#output: Image with feature points projected
def projectFeaturePoints(img, featMask, descriptor):
    #Extract image shape
    imgHeight, imgWidth = img.shape[:2]
    
    #Copy image to allow alteration
    imgAlter = img.copy()
    
    #Load image into MatPlot
    plt.imshow(imgAlter)
    plt.axis('off')

    #Iterate through image and add feature points
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (featMask[y][x] == 255):
                plt.plot(x, y, marker = 'x', color = 'yellow')
              
    #Save modified image then close plot
    plt.savefig(outPath + "FeaturePoints" + descriptor + ".jpg", dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()

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

#Function to detect corners using Harris-Stephens Detector and non-maximum suppression
#Input: Auto-correlation matrix of input image
#Output: Array of R values for whole image
def computeFeatureResponseNMS(img, matrix):
    #Set value of empirical constant [0.04, 0.06]
    alpha = 0.05
    
    #Apply NMS to tensor determinant and matrix
    tensorDet = nonMaxSuppression(img, tensorDeterminant(matrix))
    tensorTrc = nonMaxSuppression(img, tensorTrace(matrix[0], matrix[3]))
    
    return np.subtract(tensorDet, np.multiply(alpha, np.power(tensorTrc, 2)))

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
        
    #Extract image size
    imgHeight, imgWidth = R.shape[:2]
    
    #Connect feature coordinates to R-value
    featCoorVal = []
    for x in range(len(featCoor)):
        #Ensure feature is far enough away from image borders to be represented with SIFT
        if (featCoor[x][0] > 8 and featCoor[x][0] < imgWidth - 8 and featCoor[x][1] > 8 and featCoor[x][1] < imgHeight - 8):
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
def generateFeatureDescriptorsANT(img, featCoor, rad):
    #Determine npoints from featCoor length
    npoints = len(featCoor)
    
    #Initialize array to hold descriptor list
    Dlist = []
    
    #Extract image size
    imgHeight, imgWidth = img.shape[:2]
    
    #Iterate through featCoor list and extract descriptor for each
    for i in range(npoints):
        #Declare array to hold 5x5 RGB image of descriptor
        descriptor = np.zeros((1 + rad * 2, 1 + rad * 2,3), dtype=np.int16)
        
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

#Function to generate feature descriptors from list of feature points
#Input: Original image and list of feature coordinates
#Output: List of feature descriptors, indexed identically to the input feature coordinates
def generateFeatureDescriptors(img, featCoor):
    #Determine npoints from featCoor length
    npoints = len(featCoor)
    
    #Initialize array to hold descriptor list
    Dlist = []
    
    #Get gradient magnitude and orientation
    gradMagnitude, gradOrient = gradMagOrHelper(img)
    
    #Iterate through featCoor list and extract descriptor for each
    for i in range(npoints):
        #Declare array to hold descriptor (4x4 for smaller sections, and 8 for bins in HoG)
        #Bin index correlates with its orientaiton, increasing by pi/4 from 0 - 2pi
        desc = np.zeros((4,4,8), dtype=np.float64)
        
        #For loops to iterate through each array subsection
        for subY in range(4):
            for subX in range(4):
                #For loops to iterate through 4x4 sample contained in each subsection
                for sampY in range(4):
                    for sampX in range(4):
                        #Get pixel magnitude and orientation for sample location
                        pixelMag = gradMagnitude[featCoor[i][0] - 8 + (4 * subY) + sampY][featCoor[i][1] - 8 + (4 * subX) + sampX]
                        pixelOrient = int(np.round(np.divide(gradOrient[featCoor[i][0] - 8 + (4 * subY) + sampY][featCoor[i][1] - 8 + (4 * subX) + sampX], np.pi / 4)))
                        
                        #Adjust rounding to fit into bins
                        if (pixelOrient == 8):
                            pixelOrient = 7
                        
                        #Increase bin of appropriate orientation by the pixel's gradient magnitude
                        desc[subY][subX][pixelOrient] += pixelMag
                        
        #Add descriptor to Dlist
        Dlist.append(desc)
        
    #Return list of descriptors
    return Dlist

#Function to calculate L1 Norm distance between two descriptors
#Input: Two descriptors
#Output: L1 Norm
def l1Norm(desc1, desc2):
    #Initialize variable to hold sum of differences between pixels
    distSum = np.zeros(1, dtype=np.int64)
    
    #Extract descriptor size
    imgHeight, imgWidth = desc1.shape[:2]
    
    #Iterate through descriptors and calculate individual pixel distance, than add to running sum
    for y in range(imgHeight):
        for x in range(imgWidth):
            distSum += np.abs(desc1[y][x][0] - desc2[y][x][0])
            distSum += np.abs(desc1[y][x][1] - desc2[y][x][1])
            distSum += np.abs(desc1[y][x][2] - desc2[y][x][2])
            
    #Return L1 Norm
    return distSum

#Function to calculate Euclidian distance between two SIFT descriptors
#Input: Two descriptors
#Output: Euclidian distance
def euclidianDistance(desc1, desc2):
    #Initialize variable to hold sum of distances between each descriptor subsection
    totalDist = np.zeros(1, dtype=np.float64)
    
    #Iterate through 4x4 subsection of descriptors to calculate distance, then add to sum
    for i in range(4):
        for j in range(4):
            #Sum squared difference for each histogram bin
            squarDiffSum = np.zeros(1, dtype=np.float64)
            for k in range(8):
                squarDiffSum += np.pow(desc1[i][j][k] - desc2[i][j][k], 2)
                
            #Add square root to distance sum
            totalDist += np.sqrt(squarDiffSum)
            
    #Return total distance
    return totalDist

#Funcrion to determine dominant horizontal/vertical edge angle to estimate transformation
#Input: Image gradient orientation, magnitude, and magnitude threshold for inclusion in calculations
#Output: Dominant horizontal and vertical angle (in radians from 0 - 2pi)
def determineDominantAngles(gradOrient, gradMagnitude, threshold):
    #Extract image size
    imgHeight, imgWidth = gradOrient.shape[:2]
    
    #Create arrays to hold vert/horiz histogram
    vertHist = np.zeros(12, dtype=np.float64)
    horizHist = np.zeros(12, dtype=np.float64)
    
    #Iterate through image positions and add magnitude to appropriate bin
    for y in range(imgHeight):
        for x in range(imgWidth):
            #Ensure magnitude is greater than threshold
            if (gradMagnitude[y][x] > threshold):
                pixelOrient = np.divide(gradOrient[y][x] + (2 * np.pi), np.pi / 6)
                
                #Add magnitude into appropriate bin (1-6 vert are pi/4 - 3pi/4, 1-6 horiz are 3pi/4 - 5pi/4, 7-12 vert are 5pi/4 - 7pi/4, 7-12 vert are 7pi/4 - pi/4)
                if (gradOrient[y][x] >= np.pi / 4 and gradOrient[y][x] < 3 * np.pi / 4):
                    pixelBin = int(np.round(pixelOrient - 1))
                    vertHist[pixelBin % 6] += gradMagnitude[y][x]
                    
                elif (gradOrient[y][x] >= 3 * np.pi / 4 and gradOrient[y][x] < 5 * np.pi / 4):
                    pixelBin = int(np.round(pixelOrient - 3))
                    horizHist[pixelBin % 6] += gradMagnitude[y][x]
                    
                elif (gradOrient[y][x] >= 5 * np.pi / 4 and gradOrient[y][x] < 7 * np.pi / 4):
                    pixelBin = int(np.round(pixelOrient - 5))
                    vertHist[5 + (pixelBin % 6)] += gradMagnitude[y][x]
                    
                else:
                    pixelBin = int(np.round(pixelOrient - 7))
                    horizHist[5 + (pixelBin % 6)] += gradMagnitude[y][x]
                    
    #Determine maximum index of histogram
    maxVert = 0
    maxHoriz = 0
    maxVertInd = np.argmax(vertHist)
    maxHorizInd = np.argmax(horizHist)
    
    
    if (maxVertInd < 6):
        maxVert = (maxVertInd + 1) * (np.pi / 6)
    else:
        maxVert = (5 * np.pi / 4) + (maxVertInd - 5) * (np.pi / 6)
        
    if (maxHorizInd < 6):
        maxHoriz = (maxHorizInd + 1) * (np.pi / 6)
    elif (maxHorizInd < 9):
        maxHoriz = (5 * np.pi / 4) + (maxHorizInd - 5) * (np.pi / 6)
    else:
        maxHoriz = (maxVertInd - 9) * (np.pi / 6)
        
    #Return dominant angles
    return maxVert, maxHoriz
                

#ANTIQUATED Function to calculate descriptor distances between two image descriptor lists
#Input: Dlist of image 1 and Dlist of image 2
#Output: Matrix, where Dist(i,j) is the distance between the ith descriptor of image 1 and the jth descriptor of image 2
#TO-DO: TAILOR TO NEW DESCRIPTOR GENERATOR
def computeDescriptorDistancesANT(Dlist1, Dlist2):
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

#Function to calculate descriptor distances between two image descriptor lists
#Input: Dlist of image 1 and Dlist of image 2
#Output: Matrix, where Dist(i,j) is the distance between the ith descriptor of image 1 and the jth descriptor of image 2
def computeDescriptorDistances(Dlist1, Dlist2):
    #Initialize matrix to hold distances
    Dist = []
    
    #Iterate through Dlists and calculate distances
    for i in range(len(Dlist1)):
        #Initialize list to hold distances from Dlist1[i]
        iDist = []
        
        for j in range(len(Dlist2)):
            #Append distance from Dlist2[j]
            iDist.append(euclidianDistance(Dlist1[i], Dlist2[j]))
            
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

#Function to determine match based on distance threshold of every descriptor across images
#Input: Distance matrix and threshold value
#Output: List of matches
def Distance2Matches_DistThresh(Dist, Th1):
    #Initialize array to hold matches
    matchList = []
    
    #Iterate through distance matrix and determine match for each feature descriptor
    for i in range(len(Dist)):
        for j in range(len(Dist[i])):
            #Check if match distance is less than threshold
            if (Dist[i][j] < Th1):
                #Properly format match then append to match list
                iMatch = []
                iMatch.append(i)
                iMatch.append(j)
                
                matchList.append(iMatch)
        
    #Return list of matches
    return matchList

#Function to determine matches based on nearest neighbor ratio for every descriptor across images
#Input: Distance matrix and threshold value
#Output: List of matches
def Distance2Matches_NearestRatio(Dist, Th3):
    #Initialize array to hold matches
    matchList = []
    
    #Iterate through distance matrix and determine match for each feature descriptor
    for i in range(len(Dist)):
        #Extract two nearest neighbor distances
        min = np.min(Dist[i])
        minIndex = int(np.argmin(Dist[i]))
        
        distTemp = np.delete(Dist[i], minIndex)
        secMin = np.min(distTemp)
        
        #Ensure ratio is within threshold
        if (min / secMin < Th3):
            #Declare array to hold individual match
            iMatch = []
            iMatch.append(i)
        
            #Append index of minimum match
            iMatch.append(minIndex)
        
            #Append match for i to output list
            matchList.append(iMatch)
        
    #Return list of matches
    return matchList

#Function to read transformation matrix file to array
#Input: Path to transformation matrix
#Output: 3x3 array of float values
def readTransMatFile(transMatPath):
    #Open and read-in transformation matrix file
    transMatFile = open(transMatPath, 'r')
    transMatCont = transMatFile.read()

    #Correctly format transformation matrix into a list
    transMatCont = transMatCont.replace('  ', ',')
    transMatCont = transMatCont.replace('\n', ',')
    transMatCont = transMatCont.replace(' ', '')
    transMatListTemp = transMatCont.split(",")
    
    #Remove all empty entries
    transMatList = []
    for x in transMatListTemp:
        if (x != ''):
            transMatList.append(x)
    
    #Initialize and populate array to hold transformation matrix values for each line
    transMat = []
    
    transMatL = []
    transMatL.append(float(transMatList[0]))
    transMatL.append(float(transMatList[1]))
    transMatL.append(float(transMatList[2]))
    transMat.append(transMatL)
    
    transMatL = []
    transMatL.append(float(transMatList[3]))
    transMatL.append(float(transMatList[4]))
    transMatL.append(float(transMatList[5]))
    transMat.append(transMatL)
    
    transMatL = []
    transMatL.append(float(transMatList[6]))
    transMatL.append(float(transMatList[7]))
    transMatL.append(float(transMatList[8]))
    transMat.append(transMatL)
    
    #Return transformation matrix
    return transMat

#Function to calculate pixel distance between estimated position and matched point's position
#Input: Original point, matched position, and transformation matrix
#Output: Pixel distance between estimated position and matched point's position
def estimatedPointError(origCoor, matchCoor, transMat):
    #Calculate estimated coordinate based on transformation matrix
    estimatedCoor = np.ones((3))
    estimatedCoor[0] = origCoor[1] * transMat[0][0] + origCoor[0] * transMat[0][1] + transMat[0][2]
    estimatedCoor[1] = origCoor[1] * transMat[1][0] + origCoor[0] * transMat[1][1] + transMat[1][2]
    estimatedCoor[2] = origCoor[1] * transMat[2][0] + origCoor[0] * transMat[2][1] + transMat[2][2]
    
    #print(f'Orig: {origCoor}')
    #print(f'Estimated: {estimatedCoor}')
    
    #Return pixel distance between matched coordinate and estimated coordinate
    return np.sqrt(np.pow(estimatedCoor[1] - matchCoor[0], 2) + np.pow(estimatedCoor[0] - matchCoor[1], 2))

#Function to calculate pixel distance between all matches
#Input: matchList, featCoor1, featCoor2, and transMat
#Output: List of pixel distances indexed like MatchList
def matchListPointDistances(matchList, featCoor1, featCoor2, transMat):
    #Initialize array to hold output
    pointDistances = []
    
    #Iterate through matchList to calculate all pixel distances
    for i in range(len(matchList)):
        pointDistances.append(estimatedPointError(featCoor1[matchList[i][0]], featCoor2[matchList[i][1]], transMat))
        
    #Return error list
    return pointDistances
    
#Function to count true positives for matchlist
#Input: Point distances and true positive threshold
#Output: Number of true positives
def truePositiveCount(pointDistances, tpThresh):
    #Count number of pixel distances less than the true positive threshold
    tpCount = 0
    for x in pointDistances:
        if (x < tpThresh):
            tpCount += 1
            
    #Return true positive count
    return tpCount

#Function to plot original descriptor and matched descriptor
#Input: Original image, matched image, matchList, featCoor1, and featCoor2
#Output: None
def plotMatches(img1, img2, matchList, featCoor1, featCoor2, outPath, descriptor):
    #Initialize new image to hold output and set each side equal to an initial image
    imgHeight, imgWidth = img2.shape[:2]
    imgMatch = np.zeros((imgHeight, 2 * imgWidth, 3), dtype=np.uint8)
    for y in range(imgHeight):
        for x in range(imgWidth):
            imgMatch[y][x] = img1[y][x]
            imgMatch[y][imgWidth + x] = img2[y][x]
            
            
    #Load new image into MatPlot
    plt.imshow(imgMatch)
    plt.axis('off')
    
    #For every match, plot the descriptor points and a line connecting them
    for i in range(len(matchList)):
        xCoors = []
        xCoors.append(featCoor1[matchList[i][0]][1])
        xCoors.append(imgWidth + featCoor2[matchList[i][1]][1])
        
        yCoors = []
        yCoors.append(featCoor1[matchList[i][0]][0])
        yCoors.append(featCoor2[matchList[i][1]][0])
        
        plt.plot(featCoor1[matchList[i][0]][1], featCoor1[matchList[i][0]][0], marker='x', color='yellow')
        plt.plot(imgWidth + featCoor2[matchList[i][1]][1], featCoor2[matchList[i][1]][0], marker='x', color='yellow', linewidth=1)
        plt.plot(xCoors, yCoors, color="yellow")
        
    #Save image
    plt.savefig(outPath + "Matches" + descriptor + ".jpg", dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()
    
#Function to calculate neighboring pixels based on interpolation
#Input: Coordinates of pixel, edge image (from tensor trace), gradient magnitude image, and gradient orienation imag
#Output: Values of interpolated pixel neighbors
def interpolateNeighboringPixels(qY, qX, imgEdge, gradMagnitude, gradOrient):
    #Get central pixel gradient magnitude and orientation
    qOr = gradOrient[qY][qX]
    
    #Calculate directional distances to succeeding pixel
    yDiff = np.sin(qOr)
    xDiff = np.cos(qOr)
    
    #Determine neighboring pixel locations
    pY = qY + yDiff
    pX = qX + xDiff
    
    rY = qY - yDiff
    rX = qX - xDiff
    
    #Determine neighboring pixel edge values by interpolating from three nearest pixels (unless p & r align with real pixel)
    if (yDiff != 1 and yDiff != -1):
        pixelP = interpolatePixelValue (pY, pX, imgEdge, gradMagnitude, gradOrient)
        pixelR = interpolatePixelValue (rY, rX, imgEdge, gradMagnitude, gradOrient)
    else:
        pixelP = imgEdge[int(pY)][int(pX)]
        pixelR = imgEdge[int(rY)][int(rX)]
    
    #Return neighboring pixel values
    return pixelP, pixelR
    
#Function to get pixel distance between coordinates
#Input: X and Y coordinates of two points
#Output: Pixel distance between points
def pixelDistance(y1, x1, y2, x2):
    return np.sqrt(np.pow(y1 - y2, 2) + np.pow(x1 - x2, 2))
    
#Function to determine interpolated value of unaligned pixel
#Input: Coordinates of pixel, edge image, gradient magnitude, and gradient orientation
#Output: Interpolated pixel value
def interpolatePixelValue(y, x, imgEdge, gradMagnitude, gradOrient):
    #Get positions of bounding four pixels
    tlY = int(np.ceil(y))
    tlX = int(np.floor(x))
    
    trY = int(np.ceil(y))
    trX = int(np.ceil(x))
    
    brY = int(np.floor(y))
    brX = int(np.ceil(x))
    
    blY = int(np.floor(y))
    blX = int(np.floor(x))
    
    #Calculate distance from each bounding pixel
    tlDist = pixelDistance(y, x, tlY, tlX)
    trDist = pixelDistance(y, x, trY, trX)
    blDist = pixelDistance(y, x, blY, blX)
    brDist = pixelDistance(y, x, brY, brX)
    
    #Calculate estimated value based on each bounding pixel's gradMag and gradOrient
    tlEst = imgEdge[tlY][tlX] + (np.cos(gradOrient[tlY][tlX]) * gradMagnitude[tlY][tlX] * (tlY - y)) + (np.sin(gradOrient[tlY][tlX]) * gradMagnitude[tlY][tlX] * (x - tlX))
    trEst = imgEdge[trY][trX] + (np.cos(gradOrient[trY][trX]) * gradMagnitude[trY][trX] * (trY - y)) + (np.sin(gradOrient[trY][trX]) * gradMagnitude[trY][trX] * (trX - x))
    brEst = imgEdge[brY][brX] + (np.cos(gradOrient[brY][brX]) * gradMagnitude[brY][brX] * (y - brY)) + (np.sin(gradOrient[brY][brX]) * gradMagnitude[brY][brX] * (x - brX))
    blEst = imgEdge[blY][blX] + (np.cos(gradOrient[blY][blX]) * gradMagnitude[blY][blX] * (y - blY)) + (np.sin(gradOrient[blY][blX]) * gradMagnitude[blY][blX] * (blX - x))
    
    #Calculate interpolated value by weighted average based on inverse distance to bounding pixels
    totDistPrelim = tlDist + trDist + blDist + brDist
    tlDist = 1 / (tlDist / totDistPrelim)
    trDist = 1 / (trDist / totDistPrelim)
    blDist = 1 / (blDist / totDistPrelim)
    brDist = 1 / (brDist / totDistPrelim)
    totDist = tlDist + trDist + blDist + brDist
    
    return (tlEst * (tlDist / totDist)) + (trEst * (trDist / totDist)) + (brEst * (brDist / totDist)) + (blEst * (blDist / totDist))
        
#Function to perform non-maximum suppression
#Input: Image
#Output: Edge image with non-maximally suppressed lines
def nonMaxSuppression(img, imgEdge):
    #Extract image size information
    imgHeight, imgWidth = imgEdge.shape[:2]
    
    #Get gradient magnitude and orientation images
    gradMagnitude, gradOrient = gradMagOrHelper(img)
    
    #Iterate through every internal pixel and discard if not max along gradient
    for y in range(1, imgHeight - 1):
        for x in range(1, imgWidth - 1):
            #Get current pixel edge value
            pixelQ = imgEdge[y][x]
            
            #Check if pixel is edge, otherwise skip iteration
            if (pixelQ == 0):
                continue
            
            #Determine value of preceeding and succeeding pixels
            pixelP, pixelR = interpolateNeighboringPixels(y, x, imgEdge, gradMagnitude, gradOrient)
            
            #Set to 0 if not max pixel
            if (pixelQ < pixelP or pixelQ < pixelR):
                imgEdge[y][x] = 0
                
    #Return max-only edge image
    return imgEdge
             


#Get paths for input images, output directory, and small/large-scale sigma values
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath1 = str(srcDir + '\\' + sys.argv[1])
inPath2 = str(srcDir + '\\' + sys.argv[2])
outPath = str(srcDir + '\\' + sys.argv[3])
transMatPath = str(srcDir + '\\' + sys.argv[4])
sigma1 = float(sys.argv[5])
sigma2 = float(sys.argv[6])
npoints = int(sys.argv[7])

#Create output directory if nonexistent
if (not os.path.exists(outPath)):
    os.mkdir(outPath)


#Read in input image
img1 = cv.imread(inPath1)
img2 = cv.imread(inPath2)

'''gradMagnitude1, gradOrient1 = gradMagOrHelper(img1)
gradMagnitude2, gradOrient2 = gradMagOrHelper(img2)

maxVert1, maxHoriz1 = determineDominantAngles(gradOrient1, gradMagnitude1, 0)
maxVert2, maxHoriz2 = determineDominantAngles(gradOrient2, gradMagnitude2, 0)

print(maxVert1 / (np.pi / 6))
print(maxHoriz1/ (np.pi / 6))
print(maxVert2/ (np.pi / 6))
print(maxHoriz2/ (np.pi / 6)) '''

#Create arrays to hold output images and their filenames, and add input
imgOutputs = []
imgNames = []

imgOutputs.append(img1)
imgNames.append("Input1")
imgOutputs.append(img2)
imgNames.append("Input2")


#Generate auto-correlation matrices
print("Generating auto-correlation matrices...")
matrix1 = genAutoCorrelationMatrix(img1, sigma1)
matrix2 = genAutoCorrelationMatrix(img2, sigma2)


#Calculate feature response and save to output
print("Calculating feature response r-values...")
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
print("Calculating local maxima...")
featCoor1, featCoorMask1 = featureMeasure2Points(rVals1, npoints)
featCoor2, featCoorMask2 = featureMeasure2Points(rVals2, npoints)

imgOutputs.append(featCoorMask1)
imgNames.append("FeaturePointMask1")
projectFeaturePoints(img1, featCoorMask1, "1")


imgOutputs.append(featCoorMask2)
imgNames.append("FeaturePointMask2")
projectFeaturePoints(img2, featCoorMask2, "2")


#Generate list of descriptors, then matrix of distances
print("Generating feature descriptors...")
Dlist1 = generateFeatureDescriptors(img1, featCoor1)
Dlist2 = generateFeatureDescriptors(img2, featCoor2)
Dist = computeDescriptorDistances(Dlist1, Dlist2)


#Perform matching between descriptor lists
print("Matching between descriptor lists...")
matchList1 = Distance2Matches_DistThresh(Dist, 40000)
matchList2 = Distance2Matches_NearestMatch(Dist, 40000)
matchList3 = Distance2Matches_NearestRatio(Dist, .9)


#Read in transformation matrix file
print("Comparing to estimated values from transformation matrix...")
transMat = readTransMatFile(transMatPath)
matchList1PD = matchListPointDistances(matchList1, featCoor1, featCoor2, transMat)
matchList2PD = matchListPointDistances(matchList2, featCoor1, featCoor2, transMat)
matchList3PD = matchListPointDistances(matchList3, featCoor1, featCoor2, transMat)

plotMatches(img1, img2, matchList1, featCoor1, featCoor2, outPath, "DistanceThreshold")
plotMatches(img1, img2, matchList2, featCoor1, featCoor2, outPath, "NearestMatch")
plotMatches(img1, img2, matchList3, featCoor1, featCoor2, outPath, "NearestRatio")


#Calculate true positive rate
print(f'For distance threshold, number correct out of {len(matchList1PD)} matches: {truePositiveCount(matchList1PD, 5)}')
print(f'For nearest match, number correct out of {len(matchList2PD)} matches: {truePositiveCount(matchList2PD, 5)}')
print(f'For nearest neighbor ratio, number correct out of {len(matchList3PD)} matches: {truePositiveCount(matchList3PD, 5)}')


#Save results to text file
resultsText = open(outPath + "Results.txt", "a")
resultsText.write(f'For distance threshold, number correct out of {len(matchList1PD)} matches: {truePositiveCount(matchList1PD, 5)}\n')
resultsText.write(f'For nearest match, number correct out of {len(matchList2PD)} matches: {truePositiveCount(matchList2PD, 5)}\n')
resultsText.write(f'For nearest neighbor ratio, number correct out of {len(matchList3PD)} matches: {truePositiveCount(matchList3PD, 5)}\n')
resultsText.close()


#Save all output images
saveImages(outPath, imgOutputs, imgNames)