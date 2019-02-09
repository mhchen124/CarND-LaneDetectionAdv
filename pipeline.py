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

    def __init__(self):
        # current line detection flag
        self.current_detected = False
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.current_curverad = None
        #distance in meters of vehicle center from the line
        self.current_line_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

class HighwayLane():

    # Number of previous fits we remember
    N = 3
    f1 = (N - 1) / N
    f2 = 1 / N

    # was the line detected in the last iteration?
    prev_detected = False
    # average curvature of last n
    prev_curverad = 1000
    # average line position of last n
    prev_line_pos = 2
    # x values of the last n fits of the line
    left_lastn_fittedx = []
    right_lastn_fittedx = []
    # average x values of the fitted line over the last n iterations
    left_lastn_bestx = None
    right_lastn_bestx = None
    # polynomial coefficients averaged over the last n iterations
    left_lastn_best_fit = None
    right_lastn_best_fit = None

    left_line = None
    right_line = None

    def __int__(self):
        pass

    def __int__(self, left, right):
        self.left_line = left
        self.right_line = right

    def set_left_line(self, l):
        self.left_line = l
    def set_right_line(self, r):
        self.right_line = r

    def update_curverad(self, curverad):
        self.prev_curverad = self.f1 * self.prev_curverad + self.f2 * curverad

    def update_position(self, pos):
        self.prev_line_pos = self.f1 * self.prev_line_pos + self.f2 * pos

    def update_left_best_fit(self, fit):
        self.lastn_best_fit = self.f1 * self.lastn_best_fit + self.f2 * fit


def image_precessor1(img):

    img_row = img.shape[0]
    img_col = img.shape[1]

    if DEBUG:
        print("img_row: ", img_row)
        print("img_col: ", img_col)

    left_lane = Line()
    right_lane = Line()
    my_lane = HighwayLane()
    my_lane.set_left_line(left_lane)
    my_lane.set_right_line(right_lane)

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
    #clip1 = clip1.subclip(0, 5)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(out_video_file, audio=False)


#run_image_pipe(INPUT_DIR)

run_video_pipe(read_in, white_output)
