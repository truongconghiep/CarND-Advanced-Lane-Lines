# Advanced land finding project

**Hiep Truong Cong**

[//]: # (Image References)
[image1]: ./output_images/Chessboard_corner.png "Chessboard corners"
[image2]: ./output_images/Distortion_Correction.jpg "Distortion Correction"
[image3]: ./output_images/WhiteAndYellowMasking.jpg "White and yellow masking"
[image4]: ./output_images/binary_thresholding.jpg "Binary thresholding"
[image5]: ./output_images/Direction_Gradient_Thresholding.jpg "Direction gradient"
[image6]: ./output_images/Combined_Thresholding.jpg "Combined thresholding"
[image7]: ./output_images/Perspective_Transformation.jpg "Perspective transformation"
[image8]: ./output_images/PL_Img_Distortion_Correction.jpg "Distortion Correction in pipeline"
[image9]: ./output_images/PL_Thresholding.jpg "Thresholding in pipeline"
[image10]: ./output_images/PL_Perspective_Transform.jpg "Perspective transformation in pipeline"
[image11]: ./output_images/PL_searching_window.jpg "Searching window technique"
[image12]: ./output_images/PL_Margin_Visualization.jpg "Margin visualization"
[image13]: ./output_images/PL_final_image.jpg "Final image"
[image7]: ./output_images/
[image7]: ./output_images/

[video1]: ./project_video.mp4 "Video"

---

## Submitted files

  * [writeup](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND-Advanced-Lane-Lines-writeup.md) you are reading it
  * [Jupyter Notebook](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND-Advanced-Lane-Lines.ipynb) and [python source](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py)
  * [Example output image](./output_images/PL_final_image.jpg)
  * [Example output videos](https://www.youtube.com/watch?v=BPpeH9Xzc-0) and [challenge video](https://www.youtube.com/watch?v=kJhs2Iq-Q6o)
  
## The goals / steps of this project

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
  * Find object points and image points 
    + Iterate through all calibration images, provided by Udacity
    + For each image calculate apply cv2.findChessboardCorners to find its object points (corners) and image points
    +  register these object and image points for the next step
  * Apply cv2.calibrateCamera for the found object points and image points to calculate image matrix and distortion coefficients.
  * Apply the `cv2.undistort()` function to correct the distortion on the image
This algorithm is implemeted in the function [*camera_calibration*](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L58). In the following picture, found chessboard corners on an image are shown
![alt text][image1]

## Distortion correction
After the image matrix and distortion coefficients are calculated, the distortion will be corrected by applying the ["undistort_img"](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L76) function. A distortion corrected data is shown below
![alt text][image2]

## Color and gradient thresholding

Color and gradient thresholding (see function ["color_Gradient_Threshold"](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L146))
  * Color filtering (see function ["MaskYellowAndWhite"](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L112))
     + Select yellow pixels in RGB color space
     + Select white pixels in RGB color space
     + Select yellow pixels in HLS color space
  * Gradient thresholding: direction gradient is applied to find out edges in the original image
  * Combine color and gradient to select expected pixels from the image
      ![alt text][image3]
      ![alt text][image4]
      ![alt text][image5]
      ![alt text][image6]

## Perspective transformation
In this step the thresholded image from previous step will be transformed in bird-eye perspective. Images in this perspective reflect  shapes of objects in real world, with it lanelines parameter such as curvatures will be easy determined. A perspective transformation is performed in following steps:
   * Determine transformation matrix
      + Determine source points on the original image and destination points on road surface.
      + Call function cv2.getPerspectiveTransform to get the transformation matrix. This function takes some point of source image and some of destination image as input. These points are chosen as below

           | Source (y,x)  | Destination(y,x)|
           |:-------------:|:---------------:|
           | 273, 672      | 273, 720        |
           | 570, 466      | 273, 0          |
           | 712, 466      | 1030,0          |
           | 1030, 672     | 1030, 720       |
           
   * Perform perspective transformation on the original image with the ["perspective_img_warp"](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L79) function 
An example of perspective transformation is shown in the figure below
![alt text][image7]

## Pipeline (single images) ([code here](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L463))

   1. Distortion correction
      ![alt text][image8]
   2. Perform color and gradient thresholding on the original image
      ![alt text][image9]
   3. Perform perspective transformation on the original image
      ![alt text][image10]
   4. [Finding laneline in the transformed image](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L197)
      + In this step pixels of the lanelines are determined in this [piece of code](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L234-#L26)
      + Line curvatures and the car position from the image center are also determined [here](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L282)
   5. Visualization 
      + Searching window technique [see](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L331)
      + [Margin visualization](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L353)
         ![alt text][image11]
         ![alt text][image12]
   6. [Draw detected lane in road surface space](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L392)
   7. Transforn drawnd lane back to camera perspective space
   8. Combine drawn lane image to the original image
   
      ![alt text][image13]

## Pipeline (video) ([code here](https://github.com/truongconghiep/CarND-Advanced-Lane-Lines/blob/master/CarND_Advanced_Lane_Lines.py#L420))

Here's a [link to my video result](https://www.youtube.com/watch?v=BPpeH9Xzc-0) or [here](./output_videos/project_video.mp4)

 1. Distortion correction
 2. Perform color and gradient thresholding on the original image
 3. Perform perspective transformation on the original image
 4. Finding laneline in the transformed image. Using *find_lines_in_1st_frame* for the first frame or whenever fail to detect lanelines in a frame and *find_lines* for other frame
 5. Draw detected lane in road surface space
 6. Transforn drawnd lane back to camera perspective space
 7. Add information of curvatures and offset to the frame
 
My laneline detector also works for the *challenge_video*. Here 's a link to the [challenge video](https://www.youtube.com/watch?v=kJhs2Iq-Q6o) or [here](./output_videos/challenge_video.mp4)
 

### Discussion

One of the difficulties in this project is to select correct pixels, which are representing lanelines. Due to changes of light conditions, color fading of linelines it is very difficult to filter out exactly the pixels of interess. Many techniques are available to detect lanelines, for example color thresholding and gradient thresholding, but every of them works in only some certain conditions. To build a realiable detector I have to combine these techniques together than tune their parameters, so that lanelines can be detected robustly. My detector works well on the project video and still has some flaws on the challenge video. In the future, a low-pass filter can be built in the detector to make it more stable to changes of light condition, also on roads in rough terrances. 
