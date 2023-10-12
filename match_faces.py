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

def compare(face_descriptor1, face_descriptor2):
    val = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))
    return 1 - val

def exists_combination(people, a, b):
    for people in peoples:
        if a in people and b in people:
            return True
    return False

def exists_dir(people, p):
    for people in peoples:
        for person in people:
            if person.startswith(p):
                return True
    return False

def get_path(p):
    s = p.split('/')
    return "/".join([s[len(s) - 2], s[len(s) - 1]])


file_paths_base = []
for root, _dirs, files in os.walk("faces"):
    for file in files:
        file_paths_base.append(os.path.join(root, file))

reps = dict()
file_paths = sorted(file_paths_base)
peoples = []
for a, b in itertools.combinations(file_paths, 2):
    if os.path.dirname(a) == os.path.dirname(b):
        continue

    ap = get_path(a)
    bp = get_path(b)
    if exists_combination(peoples, ap, bp):
        continue
    if exists_dir(peoples, bp):
        continue

    if a in reps:
        arep = reps[a]
    else:
        arep = getRep(a)
        reps[a] = arep

    if b in reps:
        brep = reps[b]
    else:
        brep = getRep(b)
        reps[b] = brep

    distance = compare(arep, brep)
    print("compare {} {} {}".format(ap, bp, distance))
    if distance < 0.6:
        continue

    found = False
    for people in peoples:
        if ap in people:
            people.add(bp)
            found = True
            break
        elif bp in people:
            people.add(ap)
            found = True
            break
    if not found:
        peoples.append({ap, bp})

with open("face_list.txt", mode='w') as f:
    for people in peoples:
        f.write("{}\n".format(",".join(people)))

