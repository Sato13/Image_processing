import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

os.chdir(os.path.join(os.sep, os.getcwd(), "test_images"))
orb = cv2.ORB_create(nfeatures=1000000, scoreType=cv2.ORB_FAST_SCORE)
sift = cv2.xfeatures2d.SIFT_create()

test_pics = {
    "broken_solder.png" : sift.detectAndCompute(cv2.imread("broken_solder.png" , 0), None),
    "bulbous_joint.png" : sift.detectAndCompute(cv2.imread("bulbous_joint.png" , 0), None),
    "cracked_joint.png" : sift.detectAndCompute(cv2.imread("cracked_joint.png" , 0), None),
    "flux_residues.png" : sift.detectAndCompute(cv2.imread("flux_residues.png" , 0), None),
    "incomplete_joint.png" : sift.detectAndCompute(cv2.imread("incomplete_joint.png" , 0), None),
    "joint_contam.png" : sift.detectAndCompute(cv2.imread("joint_contam.png" , 0), None),
    "lifted_pads.png" : sift.detectAndCompute(cv2.imread("lifted_pads.png" , 0), None),
    "outgas.png" : sift.detectAndCompute(cv2.imread("outgas.png" , 0), None),
    "pinblow_hole.png" : sift.detectAndCompute(cv2.imread("pinblow_hole.png" , 0), None),
    "poor_hole_fill.png" : sift.detectAndCompute(cv2.imread("poor_hole_fill.png" , 0), None),
    "poor_pen.png" : sift.detectAndCompute(cv2.imread("poor_pen.png" , 0), None),
    "poor_wetting.png" : sift.detectAndCompute(cv2.imread( "poor_wetting.png", 0), None),
    "solder_ball.png" : sift.detectAndCompute(cv2.imread("solder_ball.png" , 0), None),
    "solder_flags1.png" : sift.detectAndCompute(cv2.imread("solder_flags1.png" , 0), None),
    "solder_flags2.png" : sift.detectAndCompute(cv2.imread("solder_flags2.png" , 0), None),
    "solder_shorts.png" : sift.detectAndCompute(cv2.imread("solder_shorts.png" , 0), None),
    "solder_skip.png" : sift.detectAndCompute(cv2.imread("solder_skip.png" , 0), None),
    "sunken_joint1.png" : sift.detectAndCompute(cv2.imread("sunken_joint1.png" , 0), None),
    "sunken_joint2.png" : sift.detectAndCompute(cv2.imread("sunken_joint2.png" , 0), None),
    "sunken_joint3.png" : sift.detectAndCompute(cv2.imread( "sunken_joint3.png", 0), None)
}

test_pics_orb = {
    # "broken_solder.png" : orb.detectAndCompute(cv2.imread("broken_solder.png" , 0), None), Images commented out fail to get descriptors from orb
    "bulbous_joint.png" : orb.detectAndCompute(cv2.imread("bulbous_joint.png" , 0), None),
    "cracked_joint.png" : orb.detectAndCompute(cv2.imread("cracked_joint.png" , 0), None),
    "flux_residues.png" : orb.detectAndCompute(cv2.imread("flux_residues.png" , 0), None),
    "incomplete_joint.png" : orb.detectAndCompute(cv2.imread("incomplete_joint.png" , 0), None),
    # "joint_contam.png" : orb.detectAndCompute(cv2.imread("joint_contam.png" , 0), None),
    "lifted_pads.png" : orb.detectAndCompute(cv2.imread("lifted_pads.png" , 0), None),
    "outgas.png" : orb.detectAndCompute(cv2.imread("outgas.png" , 0), None),
    "pinblow_hole.png" : orb.detectAndCompute(cv2.imread("pinblow_hole.png" , 0), None),
    # "poor_hole_fill.png" : orb.detectAndCompute(cv2.imread("poor_hole_fill.png" , 0), None),
    "poor_pen.png" : orb.detectAndCompute(cv2.imread("poor_pen.png" , 0), None),
    "poor_wetting.png" : orb.detectAndCompute(cv2.imread( "poor_wetting.png", 0), None),
    "solder_ball.png" : orb.detectAndCompute(cv2.imread("solder_ball.png" , 0), None),
    "solder_flags1.png" : orb.detectAndCompute(cv2.imread("solder_flags1.png" , 0), None),
    "solder_flags2.png" : orb.detectAndCompute(cv2.imread("solder_flags2.png" , 0), None),
    "solder_shorts.png" : orb.detectAndCompute(cv2.imread("solder_shorts.png" , 0), None),
    "solder_skip.png" : orb.detectAndCompute(cv2.imread("solder_skip.png" , 0), None),
    "sunken_joint1.png" : orb.detectAndCompute(cv2.imread("sunken_joint1.png" , 0), None),
    "sunken_joint2.png" : orb.detectAndCompute(cv2.imread("sunken_joint2.png" , 0), None),
    "sunken_joint3.png" : orb.detectAndCompute(cv2.imread( "sunken_joint3.png", 0), None)
}

test_pics_canny = {
    # "broken_solder.png" : cv2.Canny(cv2.imread("broken_solder.png" , 0), 100, 200),
    "bulbous_joint.png" : cv2.Canny(cv2.imread("bulbous_joint.png" , 0), 100, 200),
    "cracked_joint.png" : cv2.Canny(cv2.imread("cracked_joint.png" , 0), 100, 200),
    "flux_residues.png" : cv2.Canny(cv2.imread("flux_residues.png" , 0), 100, 200),
    "incomplete_joint.png" : cv2.Canny(cv2.imread("incomplete_joint.png" , 0), 100, 200),
    # "joint_contam.png" : cv2.Canny(cv2.imread("joint_contam.png" , 0), 100, 200),
    "lifted_pads.png" : cv2.Canny(cv2.imread("lifted_pads.png" , 0), 100, 200),
    "outgas.png" : cv2.Canny(cv2.imread("outgas.png" , 0), 100, 200),
    "pinblow_hole.png" : cv2.Canny(cv2.imread("pinblow_hole.png" , 0), 100, 200),
    # "poor_hole_fill.png" : cv2.Canny(cv2.imread("poor_hole_fill.png" , 0), 100, 200),
    "poor_pen.png" : cv2.Canny(cv2.imread("poor_pen.png" , 0), 100, 200),
    "poor_wetting.png" : cv2.Canny(cv2.imread( "poor_wetting.png", 0), 100, 200),
    "solder_ball.png" : cv2.Canny(cv2.imread("solder_ball.png" , 0), 100, 200),
    "solder_flags1.png" : cv2.Canny(cv2.imread("solder_flags1.png" , 0), 100, 200),
    "solder_flags2.png" : cv2.Canny(cv2.imread("solder_flags2.png" , 0), 100, 200),
    "solder_shorts.png" : cv2.Canny(cv2.imread("solder_shorts.png" , 0), 100, 200),
    "solder_skip.png" : cv2.Canny(cv2.imread("solder_skip.png" , 0), 100, 200),
    "sunken_joint1.png" : cv2.Canny(cv2.imread("sunken_joint1.png" , 0), 100, 200),
    "sunken_joint2.png" : cv2.Canny(cv2.imread("sunken_joint2.png" , 0), 100, 200),
    "sunken_joint3.png" : cv2.Canny(cv2.imread( "sunken_joint3.png", 0), 100, 200)
}

test_pics_thresh = {
    # "broken_solder.png" : cv2.adaptiveThreshold(cv2.imread("broken_solder.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "bulbous_joint.png" : cv2.adaptiveThreshold(cv2.imread("bulbous_joint.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "cracked_joint.png" : cv2.adaptiveThreshold(cv2.imread("cracked_joint.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "flux_residues.png" : cv2.adaptiveThreshold(cv2.imread("flux_residues.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "incomplete_joint.png" : cv2.adaptiveThreshold(cv2.imread("incomplete_joint.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    # "joint_contam.png" : cv2.adaptiveThreshold(cv2.imread("joint_contam.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "lifted_pads.png" : cv2.adaptiveThreshold(cv2.imread("lifted_pads.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "outgas.png" : cv2.adaptiveThreshold(cv2.imread("outgas.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "pinblow_hole.png" : cv2.adaptiveThreshold(cv2.imread("pinblow_hole.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    # "poor_hole_fill.png" : cv2.adaptiveThreshold(cv2.imread("poor_hole_fill.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "poor_pen.png" : cv2.adaptiveThreshold(cv2.imread("poor_pen.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "poor_wetting.png" : cv2.adaptiveThreshold(cv2.imread( "poor_wetting.png", 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "solder_ball.png" : cv2.adaptiveThreshold(cv2.imread("solder_ball.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "solder_flags1.png" : cv2.adaptiveThreshold(cv2.imread("solder_flags1.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "solder_flags2.png" : cv2.adaptiveThreshold(cv2.imread("solder_flags2.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "solder_shorts.png" : cv2.adaptiveThreshold(cv2.imread("solder_shorts.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "solder_skip.png" : cv2.adaptiveThreshold(cv2.imread("solder_skip.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "sunken_joint1.png" : cv2.adaptiveThreshold(cv2.imread("sunken_joint1.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "sunken_joint2.png" : cv2.adaptiveThreshold(cv2.imread("sunken_joint2.png" , 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    "sunken_joint3.png" : cv2.adaptiveThreshold(cv2.imread( "sunken_joint3.png", 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
}

def process(img1, ratio=0.7, minMatches=5):
    A = cv2.imread(img1, 0)
    # Gets the key points of each image
    kp1, des1 = sift.detectAndCompute(A, None)
    for key, value in test_pics.items():
        B = cv2.imread(key, 0)
        kp2, des2 = value
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Matches those key points based upon the ratio to make sure it is within a certain distance
        good_points = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        #If enough matches are found, we extract the locations of matched keypoints in both the images.
        lenGP = len(good_points)
        if lenGP > minMatches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            if M is not None:
                print("%s\t%s\tMatches: %d" % (img1, key, lenGP))

def process_canny(img1, ratio=0.7, minMatches=5):
    A = cv2.Canny(cv2.imread(img1, 0), 100, 200)
    # Gets the key points of each image
    kp1, des1 = sift.detectAndCompute(A, None)
    for key, value in test_pics_canny.items():
        B = value
        kp2, des2 = sift.detectAndCompute(B, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Matches those key points based upon the ratio to make sure it is within a certain distance
        good_points = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        #If enough matches are found, we extract the locations of matched keypoints in both the images.
        lenGP = len(good_points)
        if lenGP > minMatches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            if M is not None:
                print("%s\t%s\tMatches: %d" % (img1, key, lenGP))

def process_thresh(img1, ratio=0.7, minMatches=5):
    A = cv2.adaptiveThreshold(cv2.imread(img1, 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Gets the key points of each image
    kp1, des1 = sift.detectAndCompute(A, None)
    for key, value in test_pics_thresh.items():
        B = value
        kp2, des2 = sift.detectAndCompute(B, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Matches those key points based upon the ratio to make sure it is within a certain distance
        good_points = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        #If enough matches are found, we extract the locations of matched keypoints in both the images.
        lenGP = len(good_points)
        if lenGP > minMatches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            if M is not None:
                print("%s\t%s\tMatches: %d" % (img1, key, lenGP))


def process_orb(img1, ratio=0.7, minMatches=5):
    A = cv2.imread(img1, 0)
    # Gets the key points of each image
    kp1, des1 = orb.detectAndCompute(A, None)
    for key, value in test_pics_orb.items():
        B = cv2.imread(key, 0)
        kp2, des2 = value
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        if des2 is None:
            print(key)
            break
        matches = bf.match(des1, des2)
        # Matches those key points based upon the ratio to make sure it is within a certain distance
        good_points = []
        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < 0.7 * matches[i+1].distance:
                good_points.append(m)
        #If enough matches are found, we extract the locations of matched keypoints in both the images.
        lenGP = len(good_points)
        if lenGP > minMatches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            if M is not None:
                print("%s\t%s\tMatches: %d\t" % (img1, key, lenGP))

def process_canny_orb(img1, ratio=0.7, minMatches=5):
    A = cv2.Canny(cv2.imread(img1, 0), 100, 200)
    # Gets the key points of each image
    kp1, des1 = orb.detectAndCompute(A, None)
    for key, value in test_pics_canny.items():
        B = value
        kp2, des2 = orb.detectAndCompute(B, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        if des2 is None:
            print(key)
            break
        matches = bf.match(des1, des2)
        # Matches those key points based upon the ratio to make sure it is within a certain distance
        good_points = []
        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < 0.7 * matches[i+1].distance:
                good_points.append(m)
        #If enough matches are found, we extract the locations of matched keypoints in both the images.
        lenGP = len(good_points)
        if lenGP > minMatches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            if M is not None:
                print("%s\t%s\tMatches: %d\t" % (img1, key, lenGP))

def process_thresh_orb(img1, ratio=0.7, minMatches=5):
    A = cv2.adaptiveThreshold(cv2.imread(img1, 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Gets the key points of each image
    kp1, des1 = orb.detectAndCompute(A, None)
    for key, value in test_pics_thresh.items():
        B = value
        kp2, des2 = orb.detectAndCompute(B, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        if des2 is None:
            print(key)
            break
        matches = bf.match(des1, des2)
        # Matches those key points based upon the ratio to make sure it is within a certain distance
        good_points = []
        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < 0.7 * matches[i+1].distance:
                good_points.append(m)
        #If enough matches are found, we extract the locations of matched keypoints in both the images.
        lenGP = len(good_points)
        if lenGP > minMatches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            if M is not None:
                print("%s\t%s\tMatches: %d\t" % (img1, key, lenGP))
                
process('a.jpg')
process('b.jpg')
process('c.jpg')
process('d.jpg')
process('e.jpg')
process('f.jpg')
process('g.jpg')
process('h.jpg')
process('i.jpg')
process('j.jpg')
process('k.jpg')
process('l.jpg')
process('m.jpg')
process('n.jpg')
process('o.jpg')
process('p.jpg')
process('q.jpg')
process('r.jpg')
process('s.jpg')
process('u.jpg')
process('v.png')
process('newtest.png')
process('newtest2.jpeg')
process('newtest3.jpeg')
process('newtest4.jpg')
process('newtest5.jpg')
process_canny('a.jpg')
process_canny('b.jpg')
process_canny('c.jpg')
process_canny('d.jpg')
process_canny('e.jpg')
process_canny('f.jpg')
process_canny('g.jpg')
process_canny('h.jpg')
process_canny('i.jpg')
process_canny('j.jpg')
process_canny('k.jpg')
process_canny('l.jpg')
process_canny('m.jpg')
process_canny('n.jpg')
process_canny('o.jpg')
process_canny('p.jpg')
process_canny('q.jpg')
process_canny('r.jpg')
process_canny('s.jpg')
process_canny('u.jpg')
process_canny('v.png')
process_canny('newtest.png')
process_canny('newtest2.jpeg')
process_canny('newtest3.jpeg')
process_canny('newtest4.jpg')
process_canny('newtest5.jpg')
process_thresh('a.jpg')
process_thresh('b.jpg')
process_thresh('c.jpg')
process_thresh('d.jpg')
process_thresh('e.jpg')
process_thresh('f.jpg')
process_thresh('g.jpg')
process_thresh('h.jpg')
process_thresh('i.jpg')
process_thresh('j.jpg')
process_thresh('k.jpg')
process_thresh('l.jpg')
process_thresh('m.jpg')
process_thresh('n.jpg')
process_thresh('o.jpg')
process_thresh('p.jpg')
process_thresh('q.jpg')
process_thresh('r.jpg')
process_thresh('s.jpg')
process_thresh('u.jpg')
process_thresh('v.png')
process_thresh('newtest.png')
process_thresh('newtest2.jpeg')
process_thresh('newtest3.jpeg')
process_thresh('newtest4.jpg')
process_thresh('newtest5.jpg')
process_orb('a.jpg')
process_orb('b.jpg')
process_orb('c.jpg')
process_orb('d.jpg')
process_orb('e.jpg')
process_orb('f.jpg')
process_orb('g.jpg')
process_orb('h.jpg')
process_orb('i.jpg')
process_orb('j.jpg')
process_orb('k.jpg')
process_orb('l.jpg')
process_orb('m.jpg')
process_orb('n.jpg')
process_orb('o.jpg')
process_orb('p.jpg')
process_orb('q.jpg')
process_orb('r.jpg')
process_orb('s.jpg')
process_orb('u.jpg')
process_orb('v.png')
process_orb('newtest.png')
process_orb('newtest2.jpeg')
process_orb('newtest3.jpeg')
process_orb('newtest4.jpg')
process_orb('newtest5.jpg')
process_canny_orb('a.jpg')
process_canny_orb('b.jpg')
process_canny_orb('c.jpg')
process_canny_orb('d.jpg')
process_canny_orb('e.jpg')
process_canny_orb('f.jpg')
process_canny_orb('g.jpg')
process_canny_orb('h.jpg')
process_canny_orb('i.jpg')
process_canny_orb('j.jpg')
process_canny_orb('k.jpg')
process_canny_orb('l.jpg')
process_canny_orb('m.jpg')
process_canny_orb('n.jpg')
process_canny_orb('o.jpg')
process_canny_orb('p.jpg')
process_canny_orb('q.jpg')
process_canny_orb('r.jpg')
process_canny_orb('s.jpg')
process_canny_orb('u.jpg')
process_canny_orb('v.png')
process_canny_orb('newtest.png')
process_canny_orb('newtest2.jpeg')
process_canny_orb('newtest3.jpeg')
process_canny_orb('newtest4.jpg')
process_canny_orb('newtest5.jpg')
process_thresh_orb('a.jpg')
process_thresh_orb('b.jpg')
process_thresh_orb('c.jpg')
process_thresh_orb('d.jpg')
process_thresh_orb('e.jpg')
process_thresh_orb('f.jpg')
process_thresh_orb('g.jpg')
process_thresh_orb('h.jpg')
process_thresh_orb('i.jpg')
process_thresh_orb('j.jpg')
process_thresh_orb('k.jpg')
process_thresh_orb('l.jpg')
process_thresh_orb('m.jpg')
process_thresh_orb('n.jpg')
process_thresh_orb('o.jpg')
process_thresh_orb('p.jpg')
process_thresh_orb('q.jpg')
process_thresh_orb('r.jpg')
process_thresh_orb('s.jpg')
process_thresh_orb('u.jpg')
process_thresh_orb('v.png')
process_thresh_orb('newtest.png')
process_thresh_orb('newtest2.jpeg')
process_thresh_orb('newtest3.jpeg')
process_thresh_orb('newtest4.jpg')
process_thresh_orb('newtest5.jpg')
os.chdir(os.path.split(os.getcwd())[0])