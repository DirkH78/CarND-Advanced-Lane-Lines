##ADVANCED LANE FINDING PROJECT
import numpy as np
import cv2
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import moviepy.editor as mpy

##Helper Functions
def ImportPicturesFromFolder(folder):
    filelist=os.listdir(folder)
    imglist = np.array([np.array(mpimg.imread(folder+fname)) for fname in filelist])
    return imglist

def CamCalLearn(imglist, nx=9 , ny=6, show=False):
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates
    for i in range(imglist.shape[0]):
        gray = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2GRAY) # convert to grayscale
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        
        # id corners found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
            # draw and display the corners
            if show == True:
                print("Processing chessboard corners for image No.:", i+1)
                img = cv2.drawChessboardCorners(imglist[i], (nx,ny), corners, ret)
                plt.imshow(img)
                plt.show()
                
    dstlist = np.zeros_like(imglist)
    for i in range(imglist.shape[0]):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imglist[i].shape[1], imglist[i].shape[0]) ,None,None)
        dstlist[i] = cv2.undistort(imglist[i], mtx, dist, None, mtx)
        if show == True:
            print("Processing camera calibration for image No.:", i+1)
            plt.imshow(dstlist[i])
            plt.show()
    return dstlist, imgpoints, objpoints

def UnDistImgList(imglist, imgpoints, objpoints, show=False):
    dstlist = np.zeros_like(imglist)
    for i in range(imglist.shape[0]):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imglist[i].shape[1], imglist[i].shape[0]) ,None,None)
        dst=cv2.undistort(imglist[i], mtx, dist, None, mtx)
        dstlist[i]=dst
        if show == True:
            print("Processing camera calibration for image No.:", i+1)
            plt.imshow(dst)
            plt.show()
    return dstlist

def PersTrans(imglist, src, dst, show=False):
    M = cv2.getPerspectiveTransform(src, dst)
    warpedlist = np.zeros_like(imglist)
    for i in range(imglist.shape[0]):
        warped = cv2.warpPerspective(imglist[i], M, (imglist[i].shape[1], imglist[i].shape[0]), flags=cv2.INTER_LINEAR)
        warpedlist[i]=warped
        if show == True:
            print("Processing camera calibration for image No.:", i+1)
            plt.imshow(warped, cmap='gray')
            plt.show()
    return warpedlist

def ScaleBin(img, mag_thresh=(0, 1),mode = 0):
    scaled = (img/np.max(img))        
    sbinar = np.zeros_like(scaled)
    sbinar[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    if mode == 0:
        sbinar=sbinar*scaled
    else:
        sbinar=sbinar
    return sbinar

def EdgeDetection(imglist, show=False, mag_thresh=(50, 255)):
    EdgeDetectionList = np.zeros((imglist.shape[0],imglist.shape[1],imglist.shape[2]))
    for i in range(imglist.shape[0]):
        gray = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2GRAY)
        #kernel = np.ones((20))*255
        #gray = cv2.filter2D(gray, cv2.CV_64F, kernel)
        #filter_blurred = ndimage.gaussian_filter(gray, 1)
        #alpha = 30
        #gray = gray + alpha * (gray - filter_blurred)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        #sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        #sobmag = np.sqrt(sobelx**2+sobely**2)
        #absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        red=imglist[i,:,:,0]
        hls = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2HLS)       
        S = hls[:,:,2]
        bgr = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2BGR)       
        G = bgr[:,:,1]
        R = bgr[:,:,2]
        luv = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2LUV)       
        V = luv[:,:,2]
        
        BinRed=ScaleBin(red,(0.75, 1),mode = 0)
        BinGray=ScaleBin(gray,(0.7, 1),mode = 0)
        BinSob=ScaleBin(sobelx,(0, 1),mode = 0)
        BinS=ScaleBin(S,(0.45, 1),mode = 0)
        BinG=ScaleBin(G,(0.7, 1),mode = 0)
        BinR=ScaleBin(R,(0.85, 1),mode = 0)
        BinV=ScaleBin(V,(0.75, 1),mode = 0)
        combined=ScaleBin(BinRed+BinGray+BinSob+BinS+BinG+BinR+BinV,(0.6, 1),mode = 1)
        
        EdgeDetectionList[i]=combined
        if show == True:
            print("Processing edge detection for image No.:", i+1)
            plt.imshow(EdgeDetectionList[i], cmap='gray') #, cmap='gray'
            plt.show()
    return EdgeDetectionList

#IMPORT
calimglist=ImportPicturesFromFolder('./camera_cal/')
imglist=ImportPicturesFromFolder('./test_images/')
PrVid = mpy.VideoFileClip("./project_video.mp4")

##Camera calibration
##Distortion correction
dstcallist, imgpoints, objpoints = CamCalLearn(calimglist, show=False)
dstlist = UnDistImgList(imglist, imgpoints, objpoints, show=False)

##Color/gradient threshold
EdgeDetectionList = EdgeDetection(dstlist, show=True)

##Perspective transform (first image with straight lines was used to identify points)
src = np.float32([(268,675),
                  (594,451), 
                  (1037,675), 
                  (686,451)])
dst = np.float32([(268,719),
                  (268,0),
                  (1037,719),
                  (1037,0)])   
warpedlist = PersTrans(EdgeDetectionList, src, dst, show=True)

# estimated conversion coefficient (derived by measuring the distance of road on google maps and backed by Wiki "Interstate Highway standards")
estConvCoeff = 0.0049415 #[meters per pixel]


#PrVid.write_videofile("./output_project_video.mp4")