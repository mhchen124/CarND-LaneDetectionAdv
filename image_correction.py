import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
#%matplotlib qt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

dist_pickle = {}

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:

    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.imwrite(fname.replace("camera_cal", "camera_cal_corner"), img)
        cv2.waitKey(500)

        ret, mtx, dist, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save a set of mtx/dist for later use
        if fname.__contains__("calibration3"):
            dist_pickle["mtx"] = mtx
            dist_pickle["dist"] = dist

        dst = cv2.undistort(gray, mtx, dist, None, mtx)
        cv2.imwrite(fname.replace("camera_cal", "camera_cal_undistorted"), dst)
        cv2.waitKey(500)

cv2.destroyAllWindows()
pickle.dump( dist_pickle, open( "dist_pickle.p", "wb" ) )

# Apply the image distortion function on test1.jpg
img = cv2.imread("./test_images/test3.jpg")
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite("./output_images/test3_undistorted.jpg", dst)
cv2.waitKey(500)
