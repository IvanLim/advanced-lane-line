# General imports
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Project module imports
from Preprocessing import combined_threshold, get_transform_matrices, transform_perspective
from CameraCalibration import CameraCalibration
from LaneAnalysis import LaneAnalysis

CALIBRATION_FILES = 'camera_cal/calibration*.jpg'


#############################
# Image processing pipeline #
#############################
# One-time setups and calculations
calibration = CameraCalibration(image_path=CALIBRATION_FILES, num_corners_x=9, num_corners_y=6)
laneAnalysis = LaneAnalysis()

def process_image(image):
	# Undistort the image
	image = calibration.undistort(image)

	# Preprocess the image with colour and gradient thresholding
	binary_image = combined_threshold(image)

	# Calculate our transform matrix and its inverse
	transform_matrix, transform_matrix_inverse = get_transform_matrices(image)

	# Transform the image into a top down view for analysis
	binary_top_down_image = transform_perspective(binary_image, transform_matrix)

	left_fit, right_fit = laneAnalysis.find_lines(binary_top_down_image)

	# Now that the analysis has produced the best polynomial fits 
	# for the left and right lane lines, we generate an overlay to
	# show our results
	top_down_lane_overlay = laneAnalysis.generate_top_down_lane_overlay(image, left_fit, right_fit)

	# Transform the top down overlay to the same perspective as the original image
	lane_overlay = transform_perspective(top_down_lane_overlay, transform_matrix_inverse)

	# Combine the overlay with the original image
	output = cv2.addWeighted(image, 1.0, lane_overlay, 0.3, 0)

	# Calculate the distance from the center
	output = laneAnalysis.display_stats(output, left_fit, right_fit)

	return output


##############################################
# Run the project video through the pipeline #
##############################################
video_clip = VideoFileClip('project_video.mp4')
processed_clip = video_clip.fl_image(process_image)
processed_clip.write_videofile('project_video_output.mp4', audio=False)