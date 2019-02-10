## Advanced Lane Detection Project Writeup

---



### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1a]: ./camera_cal/calibration3.jpg "Distorted Chessboard"
[image1b]: ./camera_cal_corner/calibration3.jpg "Corners"
[image1c]: ./camera_cal_undistorted/calibration3.jpg "Undistorted Chessboard"
[image2a]: ./test_images/test3.jpg "Road Image"
[image2b]: ./output_images/test3_undistorted.jpg "Road Image Undistorted"
[image3]: ./output_images/binary.jpg "Binary Example"
[image4a]: ./output_images/lines.jpg "Source Points On Input Image"
[image4b]: ./output_images/test3_bin_warped.jpg "Warped Image with Lines"
[image4c]: ./output_images/test3_bin_warped2.jpg "Warped Image"
[image5]: ./output_images/curve_lines.jpg "Warped Image"
[image6]: ./output_images/result_overlay.jpg "Overlaid On Original Image"
[video1]: ./output_images/project_video_result.mp4 "Video"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The code for camera calibaration and image distortion is in the file "image_correction.py".
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each
calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy
of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with
the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients
using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using
the `cv2.undistort()` function and obtained this result: 

![alt text][image1a]

**Original Chessboard Input Image**

![alt text][image1b]

**Detected Corners Overlaid On Chessboard Input Image**

![alt text][image1c]

**Undistorted Chessboard Image**


### Pipeline (single images)


#### 1. Provide an example of a distortion-corrected image.

To carry out this step, I used one set of mtx/dist parameters obtained from the previous calibration section.
The input image here is test3.jpg, and cv2.undistort() function was used to generate the corrected image as shown below.

![alt text][image2a]

**Original Road Image**

![alt text][image2b]

**Road Image Undistorted**

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to conduct image binarization. The code is in the file "color_gradient_thre.py".
The magnitude of gradient was used with the color s-channel to do the combined thresholding (refer to function binarize()) to generate the following binary image.

![alt text][image3]

**Binary Output Image**

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My code for perspective transformation is in "perspective.py". Four source points are manually picked - with assumption that they are the four vertices
of a rectangle viewed in 3d perception, and the bottom part is aligned with the lane lines. The four destination points are selected to
be located at the places when the rectangle is viewed as a bird-eye-view:

p1 = [600,460]\
p2 = [228,720]\
p3 = [1142,720]\
p4 = [720,460]

srcPts = np.float32([p1, p2, p3, p4])\
dstPts = np.float32(\
    [[(img_size[0] / 4), 0],\
    [(img_size[0] / 4), img_size[1]],\
    [(img_size[0] * 3 / 4), img_size[1]],\
    [(img_size[0] * 3 / 4), 0]])

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 460      | 320, 0        |
| 228, 720      | 320, 720      |
| 1142, 720     | 960, 720      |
| 720, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4a]

**Source Points On Input Image**

![alt text][image4b]

**Warped Image With Reference Rectangle**

![alt text][image4c]

**Warped Binary Image**


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

At this point, our goal is to estimate out two curve lines that represent the lane lines - using (bottom half of) the warped binary image above.
Whence these two curves are determined, the curvature and lane position will be straightforward.

The code for these tasks is in the file "lane_tracker.py" as the sliding window method is used to determine lane-line pixels. The code to pick valid lane-line
pixels are at line 80-83, the collection is done for each of the rectangle windows. Then the pixel count for each window will be checked against
pre-set threshold, if the count is above the threshold we will then use the mean of all the pixel's x position as the new x location of the window.
Otherwise we ignore those insignificant amount of pixels as they may be just noise. Finally the collected xs and ys for left and right lanes are
used to do a 2nd order polyfit as shown in code inside function "fit_polynomial()" (line 118-119). The estimated curves are shown in the image below.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This calculation is also done in file "lane_tracker.py" inside function "calculate_curvature()" after the estimated lane curves
are obtained. The lane center position is calculated line 110, inside the function of "find_lane_pixels()".
For my image the results output are:

Car position:  0.00264285714286 M\
Curvature:  943.58101011 M


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the file "lane_tracker.py" inside the function `drawing_back()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_result.mp4). The link may not work right, but the result mp4 is at: ./output_images/project_video_result.mp4

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline works well on the "project_video.mp4", kine marginally well on "challenge_video.mp4", but not exceptable on "harder_challenge_video.mp4".
The major challenge is to get a good binary image where lane-line are clear and reliable. This is kind difficult, especially for tough ones like "harder_challenge_video.mp4". If more time
are available I'd better use a-prior information to guide the tracking and reduce errors.

The initial version has some problems - the result video has a few image frames where
the lane tracking was not correct. I reworked on the code by introducing a HighwayLane
class and a Line class to help with better cross-frame performance tracking. Currently only
a simple "lane width" constraint is enforced and the result improved and there is no
more incorrect frames. If have more time, other constraints can be added to handle more complicated
lane situations.