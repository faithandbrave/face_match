import time

start = time.time()

import glob
import argparse
import cv2
import itertools
import os

import dlib

parser = argparse.ArgumentParser()
parser.add_argument('--skip_exists', action='store_true')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()

detector_file = 'mmod_human_face_detector.dat'
detector_path = os.path.join("", detector_file)
cnn_face_detector = dlib.cnn_face_detection_model_v1(detector_path)

if args.verbose:
    print("Loading the dlib models took {} seconds.".format(
        time.time() - start))


def getFaces(imgPath):
    outputDir = os.path.join("faces", os.path.splitext(os.path.basename(imgPath))[0])

    if args.skip_exists and os.path.exists(outputDir):
        return

    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + {}".format(imgPath))

    start = time.time()
    bbs = cnn_face_detector(rgbImg, 1)
    if len(bbs) == 0:
        return
    #if args.verbose:
    #    print("  + Face detection took {} seconds.".format(time.time() - start))

    index = 0
    for bb in bbs:
        rect = bb.rect
        face = bgrImg[rect.top():rect.bottom(), rect.left():rect.right()]
        w, h, _ = face.shape

        if w == 0 or h == 0:
            continue

        outputPath = "{}/{:03}.jpg".format(outputDir, index)

        if os.path.exists(outputPath):
            os.remove(outputPath)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        dst = cv2.resize(face, (96, 96))
        cv2.imwrite(outputPath, dst)
        index = index + 1

#getFaces("image_small/showcase2023_00024.jpg")
for p in sorted(list(glob.glob("image_small/*.jpg", recursive=False))):
    getFaces(p)
