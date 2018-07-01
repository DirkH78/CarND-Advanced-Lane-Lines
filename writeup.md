## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[video1]: ./project_video_output.mp4 "Video"
[image1]: ./camera_cal/calibration1.png "Original"
[image2]: ./examples/undist.jpg "Undistorted"
[image3]: ./test_images/test5.jpg "Original"
[image4]: ./examples/test5_undist.png "Undistorted"
[image5]: ./test_images/test3.jpg "Original"
[image6]: ./examples/lines_image.png "Warped"
[image7]: ./examples/par_lines.png "Lines"
[image8]: ./examples/slid_win.png "Sliding Window"
[image9]: ./examples/final.png "Final Result"
[image10]: ./examples/color_fit_lines.jpg "Curvature"


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 23-53 of the Python file located in "./AdvLaneFinding.py"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
I used the calibration parameters previously identified in cv2.undistort to undistort the image:
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 98 through 134 in "./AdvLaneFinding.py").  Here's an example of my output for this step.
The original image:
![alt text][image5]
was converted to:
![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `PersTrans()`, which appears in lines 69 through 81 in the file "./AdvLaneFinding.py". This function takes as inputs an array of images (`imglist`), as well as source (`src`) and destination (`dst`) points.  I chose the source and destination points in the following manner:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 268, 675      | 268, 719      | 
| 587, 456      | 268, 0        |
| 1037, 675     | 1037, 719     |
| 695, 456      | 1037, 0       |

I verified that my perspective transform was working as expected by making sure that the images showing a straight course will show two parallel lines that are themselfes parallelly aligned with the screen.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

By utilizing a histogramm approach, I identified the possible locations of the left and right lane. After that a sliding window algorithm ("SlidingWindow()") was needed to consecutively find lane points for both lines starting from bottom to top screen (number of sliding windows: 6; width of the windows: 150; minimum number of pixels found to recenter window: 2):

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 232 through 279 in my code in function "FinalImageProcessing()". The following idea of the udacity course was utilized:
![alt text][image10]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The example showed that different roads surfaces, contrast induced by the sun or shadows have a great impact on the result. To avoid sudden errors in the calculation, a first order delay in the curvature values or a floating average calculation might increase the stablility of the algorithm. Furthermore the whole algorithm takes a lot of computational ressources. This might lead to a unnecessary high drain of the onbouard cpu in a real vehicle.
