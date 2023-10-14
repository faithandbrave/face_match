import time

start = time.time()

import argparse
import cv2
import itertools
import os
import glob

import dlib
import numpy as np
np.set_printoptions(precision=2)

parser = argparse.ArgumentParser()
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()

if args.verbose:
    print("Loading the dlib models took {} seconds.".format(
        time.time() - start))

sp = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    shape = sp(bgrImg, dlib.rectangle(0, 0, bgrImg.shape[0], bgrImg.shape[1]))
    face_descriptor = facerec.compute_face_descriptor(bgrImg, shape)
    return face_descriptor

def get_path(p):
    s = p.split('/')
    return "/".join([s[len(s) - 2], s[len(s) - 1]])


file_paths_base = []
for root, _dirs, files in os.walk("faces"):
    for file in files:
        file_paths_base.append(os.path.join(root, file))

file_paths = sorted(file_paths_base)
for p in file_paths:
    pp = get_path(p)
    output_path = "{}.npy".format(os.path.join("vecs", pp))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    rep = np.array(getRep(p))
    np.save(output_path, rep)
    print(output_path)
