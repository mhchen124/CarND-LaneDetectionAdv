import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load(open("dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('./test_images/test3.jpg')
#img = cv2.imread('./output_images/binary.jpg')

img_size = (img.shape[1], img.shape[0])

src2 = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst2 = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

p1 = [600,460]
p2 = [228,720]
p3 = [1142,720]
p4 = [720,460]

srcPts = np.float32([p1, p2, p3, p4])
dstPts = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])


def draw_quadrangle(img, p1, p2, p3, p4, color=(0,0,255), pen_size=2):

    line_img = cv2.line(img, tuple(p1), tuple(p2), color, pen_size)
    line_img = cv2.line(line_img, tuple(p2), tuple(p3), color, pen_size)
    line_img = cv2.line(line_img, tuple(p3), tuple(p4), color, pen_size)
    line_img = cv2.line(line_img, tuple(p4), tuple(p1), color, pen_size)

    return line_img

def persp_trans(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    persp_pickle = {}
    persp_pickle["PerspM"] = M
    persp_pickle["PerspMinv"] = Minv
    pickle.dump(persp_pickle, open("persp_pickle.p", "wb"))

    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, unwarped, M


def undist_persp_trans(img, mtx, dist, src, dst):

    persp_pickle = {}

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    line_img = cv2.line(undist, tuple(p1), tuple(p2), (0,0,255), 2)
    line_img = cv2.line(line_img, tuple(p2), tuple(p3), (0,0,255), 2)
    line_img = cv2.line(line_img, tuple(p3), tuple(p4), (0,0,255), 2)
    line_img = cv2.line(line_img, tuple(p4), tuple(p1), (0,0,255), 2)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    persp_pickle["PerspM"] = M
    persp_pickle["PerspMinv"] = Minv
    pickle.dump(persp_pickle, open("persp_pickle.p", "wb"))

    warped = cv2.warpPerspective(undist, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, unwarped, line_img, undist, M

#warp, warp_inv, lines, undistorted, perspective_M = undist_persp_trans(img, mtx, dist, srcPts, dstPts)

undist = cv2.undistort(img, mtx, dist, None, mtx)
line_img = draw_quadrangle(undist, p1, p2, p3, p4)
warp, warp_inv, M = persp_trans(line_img, srcPts, dstPts)

cv2.imwrite('./output_images/lines.jpg', line_img)
cv2.imwrite('./output_images/warped.jpg', warp)
cv2.imwrite('./output_images/unwarped.jpg', warp_inv)
