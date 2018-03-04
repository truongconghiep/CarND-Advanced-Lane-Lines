'''
Created on 04.03.2018

@author: Hiep Truong
'''
from CarND_Advanced_Lane_Lines import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.image as mpimg
#if __name__ == '__main__':
img = mpimg.imread('./test_images/straight_lines1.jpg')
# Camera calibration
mtx,dist = Perform_Calibration(plot_allowed = False)
# Distortion correction
undistorted_img = Undistortion_Demo(img, mtx, dist, plot_allowed = False)
# Color and gradient thresholding
thresholded_img = color_Gradient_Threshold(img, plot_allowed = True)
# Perspective transformation
M, Minv = Perspective_Transformation(undistorted_img, plot_allowed = False)
# Pipeline for images
img_p = Pipeline_image(img, M, mtx, dist, Minv)
Plot_Images([img_p],1, title = 'Detected lane')
# Pipeline for videos
left_fit = []
right_fit = []
last_left_curverad = 0.0
last_right_curverad = 0.0
last_offset = 0.0
Is_1st_Frame = True
frame_number = 0

def process_image(image):
    global frame_number, left_fit, right_fit, last_left_curverad, last_right_curverad, last_offset, Is_1st_Frame
    left_fit, right_fit, img, Detected, new_left_curverad, new_right_curverad, new_offset = Pipeline_video(image, M, Minv,mtx, dist, 
                                                       Is_1st_Frame = Is_1st_Frame,
                                                       left_fit = left_fit,
                                                       right_fit = right_fit,
                                                       last_left_curverad = last_left_curverad,
                                                       last_right_curverad = last_right_curverad,
                                                       last_offset = last_offset)
    img = add_Text_2_img(img, 'frame: ' + str(frame_number), (10,710))
    img = add_Text_2_img(img, 'Hiep Truong', (1150,710))
    frame_number = frame_number + 1
    Is_1st_Frame = not Detected
    last_left_curverad = new_left_curverad
    last_right_curverad = new_right_curverad
    last_offset = new_offset
    return img

def Processing_video(input_video, output_video):
    clip1 = VideoFileClip(input_video)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output_video, audio=False)
    
input_project_video = './test_videos/project_video.mp4'
file_name =  input_project_video.replace('./test_videos/', '')
output_project_video = './output_videos/' + file_name

Processing_video(input_project_video, output_project_video)

input_challenge_video = './test_videos/challenge_video.mp4'
file_name =  input_challenge_video.replace('./test_videos/', '')
output_challenge_video = './output_videos/' + file_name

Processing_video(input_challenge_video, output_challenge_video)

input_harder_challenge_video = './test_videos/harder_challenge_video.mp4'
file_name =  input_challenge_video.replace('./test_videos/', '')
output_harder_challenge_video = './output_videos/' + file_name

Processing_video(input_harder_challenge_video, output_harder_challenge_video)