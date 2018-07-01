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
    # create image array
    filelist=os.listdir(folder)
    imglist = np.array([np.array(mpimg.imread(folder+fname)) for fname in filelist])
    return imglist

def ExportPicturesFromList(imglist):
    # save image array
    for i in range(imglist.shape[0]):
        mpimg.imsave('output_images/image_'+str(i)+'.jpg',imglist[i])
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
        # camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imglist[i].shape[1], imglist[i].shape[0]) ,None,None)
        dstlist[i] = cv2.undistort(imglist[i], mtx, dist, None, mtx)
        if show == True:
            print("Processing camera calibration for image No.:", i+1)
            plt.imshow(dstlist[i])
            plt.show()
    return dstlist, imgpoints, objpoints

def UnDistImgList(imglist, imgpoints, objpoints, show=False):
    # Undistort image by using previously learned camera calibration
    dstlist = np.zeros_like(imglist)
    for i in range(imglist.shape[0]):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imglist[i].shape[1], imglist[i].shape[0]) ,None,None)
        # Undistort
        dst=cv2.undistort(imglist[i], mtx, dist, None, mtx)
        dstlist[i]=dst
        if show == True:
            print("Processing camera calibration for image No.:", i+1)
            plt.imshow(dst)
            plt.show()
    return dstlist

def PersTrans(imglist, src, dst, show=False):
    # create transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warpedlist = np.zeros_like(imglist)
    for i in range(imglist.shape[0]):
        # perform perspective warp
        warped = cv2.warpPerspective(imglist[i], M, (imglist[i].shape[1], imglist[i].shape[0]), flags=cv2.INTER_LINEAR)
        warpedlist[i]=warped
        if show == True:
            print("Processing camera calibration for image No.:", i+1)
            plt.imshow(warped, cmap='gray')
            plt.show()
    return warpedlist

def ScaleBin(img, mag_thresh=(0, 1),mode = 0):
    # prepare a binary/scaled image
    scaled = (img/np.max(img))        
    sbinar = np.zeros_like(scaled)
    sbinar[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    if mode == 0:
        sbinar=sbinar*scaled
    else:
        sbinar=sbinar
    return sbinar

def EdgeDetection(imglist, show=False):
    EdgeDetectionList = np.zeros((imglist.shape[0],imglist.shape[1],imglist.shape[2]))
    for i in range(imglist.shape[0]):
        # transform images to different color spaces or use sobel
        gray = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2GRAY)
        #kernel = np.ones((20))*255
        #gray = cv2.filter2D(gray, cv2.CV_64F, kernel)
        #filter_blurred = ndimage.gaussian_filter(gray, 1)
        #alpha = 30
        #gray = gray + alpha * (gray - filter_blurred)
        sobelx = cv2.GaussianBlur(cv2.Sobel(gray, cv2.CV_64F, 1, 0),(5, 5), 0)
        #sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        #sobmag = np.sqrt(sobelx**2+sobely**2)
        #absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        red=imglist[i,:,:,0]
        #blue=imglist[i,:,:,2]
        hls = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2HLS)       
        S = hls[:,:,2]
        bgr = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2BGR)       
        G = bgr[:,:,1]
        R = bgr[:,:,2]
        luv = cv2.cvtColor(imglist[i], cv2.COLOR_RGB2LUV)       
        V = luv[:,:,2]
        
        # create binary / scaled images and scale the output to gain max. contrast
        BinRed=ScaleBin(red,(0.75, 1),mode = 0)
        ScaleRed=1.2
        BinGray=ScaleBin(gray,(0.65, 1),mode = 0)
        ScaleGray=0.5
        BinSob=ScaleBin(sobelx,(0.01, 1),mode = 0)
        ScaleSob=1.7
        BinS=ScaleBin(S,(0.35, 1),mode = 0)
        ScaleS=1.5
        BinG=ScaleBin(G,(0.6, 1),mode = 0)
        ScaleG=0.3
        BinR=ScaleBin(R,(0.65, 1),mode = 0)
        ScaleR=0.4
        BinV=ScaleBin(V,(0.7, 1),mode = 0)
        ScaleV=0.7
        # combine all this masks and create sclaed/binary image again
        combined=ScaleBin(ScaleRed*BinRed+ScaleGray*BinGray+ScaleSob*BinSob+ScaleS*BinS+ScaleG*BinG+ScaleR*BinR+ScaleV*BinV,(0.52, 1),mode = 1) #Images 0.52 video 0.41
        
        EdgeDetectionList[i]=combined
        if show == True:
            print("Processing edge detection for image No.:", i+1)
            plt.imshow(EdgeDetectionList[i], cmap='gray') #, cmap='gray'
            plt.show()
    return EdgeDetectionList

def SlidingWindow(imglist, show=False):
    left_fitLst = np.zeros((imglist.shape[0],3))
    right_fitLst = np.zeros((imglist.shape[0],3))
    for i in range(imglist.shape[0]):
        img=np.uint8(imglist[i])
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 6 #9
        # Set height of windows
        window_height = np.int(img.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 150
        # Set minimum number of pixels found to recenter window
        minpix = 2 #50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 3)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 3)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:       
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
         
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
         
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fitLst[i] = left_fit
        right_fitLst[i] = right_fit
        if show == True:
            # Generate x and y values for plotting
            ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
             
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
    return left_fitLst, right_fitLst
    
def FinalImageProcessing(orImgListimgLst, warpedImgListimgLst, left_fitLst, right_fitLst, show=False):
    finalizedImgLst = np.zeros_like(orImgListimgLst)
    for i in range(orImgListimgLst.shape[0]):
        # Generate x and y values for plotting
        img = warpedImgListimgLst[i]
        origimg = orImgListimgLst[i]
        left_fit = left_fitLst[i]
        right_fit = right_fitLst[i]
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
         
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
         
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        Minv = cv2.getPerspectiveTransform(dst, src)
        dewarped = cv2.warpPerspective(color_warp, Minv, (origimg.shape[1], origimg.shape[0])) 
        
        # Combine the result with the original image
        finalizedImgLst[i] = cv2.addWeighted(origimg, 1, dewarped, 0.3, 0)
        # Fit new polynomials to x,y in world space
        y_eval = np.max(ploty)
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # Calculating the lane center (camera is center-alligned)
        center = ((left_fitx[-1] + right_fitx[-1])/2) * xm_per_pix
 
        # Calculating the distance between the lane center and the car position
        car_position = ((origimg.shape[1])/2)* xm_per_pix
        center_distance = (center - car_position)
        
        cv2.putText(finalizedImgLst[i], 'Radius of Curvature = ' + str(np.int((left_curverad+right_curverad)/2)) + ' m', (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(finalizedImgLst[i], 'Center Lane Offset = ' + str(abs(np.int(center_distance*100))) + ' cm', (40,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if show == True:
            plt.imshow(finalizedImgLst[i])
            plt.show()
    return finalizedImgLst
    
def process_image(img):
    img=img.reshape(1,img.shape[0], img.shape[1], img.shape[2])
    
    dstlist = UnDistImgList(img, imgpoints, objpoints, show=False)
    EdgeDetectionList = EdgeDetection(dstlist, show=False)
    warpedlist = PersTrans(EdgeDetectionList, src, dst, show=False)
    left_fitLst, right_fitLst = SlidingWindow(warpedlist, show=True)
    finalizedImgLst = FinalImageProcessing(dstlist, warpedlist, left_fitLst, right_fitLst, show=True)
    
    img=finalizedImgLst.reshape(finalizedImgLst.shape[1], finalizedImgLst.shape[2], finalizedImgLst.shape[3])
    return img

#IMPORT
calimglist=ImportPicturesFromFolder('./camera_cal/')
imglist=ImportPicturesFromFolder('./test_images/')

##Camera calibration and Distortion correction
global imgpoints
global objpoints
global src
global dst
dstcallist, imgpoints, objpoints = CamCalLearn(calimglist, show=False)
dstlist = UnDistImgList(imglist, imgpoints, objpoints, show=False)

##Color/gradient threshold
EdgeDetectionList = EdgeDetection(dstlist, show=False)

##Perspective transform (first image with straight lines was used to identify points)
src = np.float32([(268,675),
                  (587,456), 
                  (1037,675), 
                  (695,456)])
dst = np.float32([(268,719),
                  (268,0),
                  (1037,719),
                  (1037,0)])   
warpedlist = PersTrans(EdgeDetectionList, src, dst, show=False)

# Define conversions in x and y from pixels space to meters
global ym_per_pix
global xm_per_pix
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

##Curvature
left_fitLst, right_fitLst = SlidingWindow(warpedlist, show=True)
finalizedImgLst = FinalImageProcessing(dstlist, warpedlist, left_fitLst, right_fitLst, show=True)
#ExportPicturesFromList(finalizedImgLst)

##Process video file
#prVid = mpy.VideoFileClip("project_video.mp4")
#processedPrVid = prVid.fl_image(process_image)
#processedPrVid.write_videofile("project_video_output.mp4", audio=False)