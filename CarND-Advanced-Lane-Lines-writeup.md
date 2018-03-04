# Advanced land finding project

**Hiep Truong Cong**

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---
Table of Contents
=================
   1. [Submitted files](## Submitted files)
   1. [The goals / steps of this project are the following](## The goals / steps of this project are the following)
   1. [Camera Calibration](## Camera Calibration)
   1. [Pipeline (single image)](## Pipeline (single image))
      * [Enhancements for future](#enhancements-for-future)
   1. [Pipeline (video)](## Pipeline (video))

---

## Submitte files

  * [writeup](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND-Advanced-Lane-Lines-writeup.md) you are reading it
  * [Code](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND-Advanced-Lane-Lines.ipynb)
  * [Example output image]()
  * [Example output videos](https://www.youtube.com/watch?v=BPpeH9Xzc-0) and [challenge video](https://www.youtube.com/watch?v=kJhs2Iq-Q6o)
  
## The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

Due to imperfection in camera design and production, taken images are usually distorted. The cause of this may come from the lens monted on the camera or asembling errors. To do object recognition we need such images, that reflect reletively exact the real world, in meaning of space and dimension, therefore as first step of this project I will look for necessary parameters to correct distorted image, before processing further. The parameters are a camera matrix and distortion coefficients. The algorithm to find these two parameters are shown below

<pre><code>
    1. Find object points and image points 
      1.1 Iterate through all calibration images, provided by Udacity
      1.2 For each image calculate apply cv2.findChessboardCorners to find its object points (corners) and image points
      1.3 register these object and image points for the next step
    2. Apply cv2.calibrateCamera for the found object points and image points to calculate image matrix and distortion coefficients.
    3. Apply the `cv2.undistort()` function to correct the distortion on the image
 </code></pre>

This algorithm is implemeted in the function *camera_calibration* in the [jupyter notebook source](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND-Advanced-Lane-Lines.ipynb). In the following picture, found chessboard corners on an image are shown
!![alt text][image1]


## Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

## Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
