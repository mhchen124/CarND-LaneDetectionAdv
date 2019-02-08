import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from color_gradient_thre    import binarize
from perspective            import persp_trans, srcPts, dstPts
from lane_tracker           import find_lane_pixels, fit_polynomial, calculate_curvature, drawing_back, xm_per_pix, ym_per_pix

# Read back persisted parameters
dist_pickle = pickle.load(open("dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

persp_pickle = pickle.load(open("persp_pickle.p", "rb"))
M = persp_pickle["PerspM"]
Minv = persp_pickle["PerspMinv"]

DEBUG = False
PERSIST_TEMP_IMGS = False

class Line():
    count = 0
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def image_precessor1(img):

    img_row = img.shape[0]
    img_col = img.shape[1]

    left_lane = Line()
    right_lane = Line()

    if DEBUG:
        print("img_row: ", img_row)
        print("img_col: ", img_col)

    if DEBUG:
        print("Undistorting input image...")
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    if PERSIST_TEMP_IMGS:
        cv2.imwrite('./output_images/temp_undist.jpg', undist)

    if DEBUG:
        print("Create binary image from undistorted image...")
    binary_img = binarize(undist)

    if PERSIST_TEMP_IMGS:
        cv2.imwrite('./output_images/temp_binary.jpg', binary_img)

    if DEBUG:
        print("Perspective transforming binary image ...")
    binary_warped, binary_unwarped, M = persp_trans(binary_img, srcPts, dstPts)

    if PERSIST_TEMP_IMGS:
        cv2.imwrite('./output_images/temp_bin_warp.jpg', binary_warped)

    if DEBUG:
        print("Detect lane & find lane boundary ...")
    out_img, leftx, lefty, rightx, righty, ploty, lane_center = find_lane_pixels(binary_warped)

    if DEBUG:
        print("Curve fitting based on lane pixels ...")
    out_img, left_fitx, right_fitx = fit_polynomial(binary_warped, leftx, lefty, rightx, righty, ploty)

    if DEBUG:
        print("Computing curvature ...")
    left_curverad, right_curverad = calculate_curvature(binary_warped, leftx, lefty, rightx, righty, ploty)

    car_pos = ((out_img.shape[1] // 2) - lane_center) * xm_per_pix
    avg_curverad = (left_curverad + right_curverad) / 2
    if DEBUG:
        print("Car position: ", car_pos, 'M')
        print("Curvature: ", avg_curverad, 'M')

    result_img = drawing_back(img, binary_warped, left_fitx, right_fitx, ploty, avg_curverad, car_pos)

    if PERSIST_TEMP_IMGS:
        cv2.imwrite('./output_images/result.jpg', result_img)

    return result_img


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    result = image_precessor1(image)
    return result

#
# Main pipeline run
#

INPUT_DIR = "./mytest/"
OUTPUT_DIR = "./temp_images/"

# Image Pipe

def LanePipe(images):
    for img_name in images:
        if DEBUG:
            print("#### Processing image ----------------> " + img_name)
        img = mpimg.imread(INPUT_DIR + img_name)
        img_result = image_precessor1(img)
        output_img_name = OUTPUT_DIR + "temp_" + img_name
        if DEBUG:
            print("Saving result image -----> " + output_img_name + "\n\n")
        mpimg.imsave(output_img_name, img_result)

def run_image_pipe(dir):
    if dir != None:
        images = os.listdir(dir)
        LanePipe(images)

# Video Pipe
read_in = './project_video.mp4'
white_output = './output_images/project_video_result.mp4'
#read_in = './challenge_video.mp4'
#white_output = './output_images/challenge_video_result.mp4'
#read_in = './harder_challenge_video.mp4'
#white_output = './output_images/harder_challenge_video_result.mp4'

def run_video_pipe(in_video_file, out_video_file):
    clip1 = VideoFileClip(in_video_file)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(out_video_file, audio=False)


#run_image_pipe(INPUT_DIR)

run_video_pipe(read_in, white_output)
