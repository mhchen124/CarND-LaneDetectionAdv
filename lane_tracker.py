import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle

# Load our image
orig = cv2.imread('./output_images/test3_undistorted.jpg')
binary_warped = mpimg.imread('./output_images/test3_bin_warped2.jpg')

ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

DEBUG = False

def find_lane_pixels(binary_warped):

    if len(binary_warped.shape) == 3:
        bin_img = binary_warped[:, :, 0]
    else:
        bin_img = binary_warped

    # Take a histogram of the bottom half of the image
    histogram = np.sum(bin_img[bin_img.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    img = np.dstack((bin_img, bin_img, bin_img))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(bin_img.shape[0] // nwindows)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = bin_img.shape[0] - (window + 1) * window_height
        win_y_high = bin_img.shape[0] - window * window_height

        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    lane_center = (leftx_base + rightx_base) / 2

    return img, leftx, lefty, rightx, righty, ploty, lane_center


def fit_polynomial(binary_warped, leftx, lefty, rightx, righty, ploty, persist=False):

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    if persist:
        plt.savefig("./output_images/fig.jpg")

    return out_img, left_fitx, right_fitx


def calculate_curvature(binary_warped, leftx, lefty, rightx, righty, ploty):

    # Adjust to real world units (meters) and redo the fitting.
    # Compute the curvature at the bottom of the road image

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    y_eval = np.max(ploty)*ym_per_pix

    sql = np.power((2*left_fit_cr[0]*y_eval + left_fit_cr[1]), 2)
    left_curverad =  np.power((1+sql), 1.5) / abs(2*left_fit_cr[0])
    sqr = np.power((2*right_fit_cr[0]*y_eval + right_fit_cr[1]), 2)
    right_curverad = np.power((1+sqr), 1.5) / abs(2*right_fit_cr[0])

    return left_curverad, right_curverad


def drawing_back(undist, binary_warped, left_fitx, right_fitx, ploty, curvature, position, persist=False):

    if DEBUG:
        print(undist.shape)
        print(binary_warped.shape)

    if len(binary_warped.shape) == 3:
        warped = binary_warped[:, :, 0]
    else:
        warped = binary_warped

    persp_pickle = pickle.load(open("persp_pickle.p", "rb"))
    Minv = persp_pickle["PerspMinv"]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Draw text on the result image
    font = cv2.FONT_HERSHEY_SIMPLEX
    UpperLeftCornerOfText1 = (10, 50)
    UpperLeftCornerOfText2 = (10, 100)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(result, 'Lane Curvature: ' + str(curvature),
                UpperLeftCornerOfText1,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.putText(result, 'Car Position (''-'' = left, ''+'' = right): ' + str(position),
                UpperLeftCornerOfText2,
                font,
                fontScale,
                fontColor,
                lineType)

    plt.imshow(result)

    if persist:
        plt.imsave("./output_images/overlay.jpg", result)

    return result

# Find our lane pixels first
out_img, leftx, lefty, rightx, righty, ploty, lane_center = find_lane_pixels(binary_warped)

# Curve fitting
out_img, left_fitx, right_fitx = fit_polynomial(binary_warped, leftx, lefty, rightx, righty, ploty)

# Compute curvature
left_curverad, right_curverad = calculate_curvature(binary_warped, leftx, lefty, rightx, righty, ploty)

# Calculation car position and avg curvature
car_pos = ((out_img.shape[1] // 2) - lane_center) * xm_per_pix
avg_curverad = (left_curverad + right_curverad) / 2

if DEBUG:
    print("Car position: ", car_pos, 'M')
    print("Curvature: ", avg_curverad, 'M')

# Drawing back on top of the original image
drawing_back(orig, binary_warped, left_fitx, right_fitx, ploty, avg_curverad, car_pos)


cv2.imwrite('./output_images/slide_win.jpg', out_img)