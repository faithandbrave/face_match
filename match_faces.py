import time

start = time.time()

import argparse
import cv2
import itertools
import os
import glob
import random

import dlib
import numpy as np
np.set_printoptions(precision=2)
random.seed(314)

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

def get_setitem(s, index):
    for i, x in enumerate(s):
        if i == index:
            return x
    return None

def exists_combination(peoples, a, b):
    for people in peoples:
        if a in people and b in people:
            return True
    return False

def exists_dir(peoples, p):
    for people in peoples:
        for person in people:
            if person.startswith(p):
                return True
    return False

def get_path(p):
    s = p.split('/')
    return "/".join([s[len(s) - 2], s[len(s) - 1]])

def make_file_paths():
    file_paths_base = []
    for root, _dirs, files in os.walk("vecs"):
        for file in files:
            file_paths_base.append(os.path.join(root, file))
    return sorted(file_paths_base)

def load_vecs(file_paths):
    vecs = dict()
    for p in file_paths:
        v = np.load(p)
        vecs[p] = v
    return vecs

def match_face_vec(face_vec1, face_vec2):
    val = np.linalg.norm(face_vec1 - face_vec2)
    return 1.0 - val

file_paths = make_file_paths()
vecs = load_vecs(file_paths)
match_cache = dict()

def match_face(path1, path2, match_rate):
    keys = sorted([path1, path2])
    key = "{}-{}".format(keys[0], keys[1])
    cached_result = match_cache.get(key)
    threshold = 0.6
    if cached_result:
        return cached_result >= threshold - match_rate

    match_result = match_face_vec(vecs[path1], vecs[path2])
    match_cache[key] = match_result
    return match_result >= threshold - match_rate

def match_list_vec(people, target_person, match_rate=0.0):
    match_count = 0
    for person in people:
        if match_face(person, target_person, match_rate):
            match_count = match_count + 1
    return match_count >= len(people) * 0.5

def match_list_list(people1, people2, match_rate=0.0):
    match_count = 0
    index1 = random.randrange(len(people1))
    if match_list_vec(people2, get_setitem(people1, index1), match_rate):
        match_count = match_count + 1

    index2 = random.randrange(len(people2))
    if match_list_vec(people1, get_setitem(people2, index2), match_rate):
        match_count = match_count + 1
    return match_count >= 2

def people_to_short_path(people):
    new_people = set()
    for person in people:
        p = get_path(person)
        new_people.add(p.replace(".npy", ""))
    return new_people

def sorted_peoples(peoples):
    return sorted(peoples, reverse=True, key=len)

# 1st step
# リストとのマッチ度が80%以上のところに入れる
def make_1st_match(paths):
    peoples = []
    for a_index, b_index in itertools.combinations(range(len(paths)), 2):
        if a_index == b_index:
            continue

        a = paths[a_index]
        b = paths[b_index]

        if os.path.dirname(a) == os.path.dirname(b):
            continue

        if exists_combination(peoples, a, b):
            continue
        if exists_dir(peoples, b):
            continue

        if not match_face(a, b, 0.0):
            continue

        found_a = False
        found_b = False
        matched = False
        for people in peoples:
            if a in people:
                found_a = True
                if match_list_vec(people, a):
                    matched = True
                    people.add(b)
                break
            elif b in people:
                found_b = True
                if match_list_vec(people, b):
                    matched = True
                    people.add(a)
                break
        if not found_a and not found_b:
            peoples.append({a, b})
        else:
            if not matched:
                if found_a:
                    peoples.append({b})
                elif found_b:
                    peoples.append({a})
    return peoples


# 2nd step
# リスト同士のマッチング
def merge_people(peoples, match_rate):
    peoples = sorted_peoples(peoples)
    disabled_indices = set()
    for a, b in itertools.combinations(range(len(peoples)), 2):
        if a == b:
            continue

        if a in disabled_indices or b in disabled_indices:
            continue

        #print("2nd compare {} / {} of {}".format(a, b, len(peoples)))
        if (len(peoples[a]) == 0 or len(peoples[b]) == 0) or match_list_list(peoples[a], peoples[b], match_rate):
            peoples[a] = peoples[a] | peoples[b]
            disabled_indices.add(b)
    for count, i in enumerate(sorted(disabled_indices)):
        del peoples[i - count]
    return peoples

# 3rd step
# リスト内のマッチ度が低い人、および人数が少ないリストを除外する
def remove_less_match_people(peoples):
    max_count = 0
    for people in peoples:
        if len(people) > max_count:
            max_count = len(people)

    new_peoples = []
    removed_people = set()
    for people in peoples:
        #print("3rd people {} / {}".format(i, len(peoples_2nd)))
        if len(people) <= int(max_count * 0.07):
            removed_people = removed_people | people
        else:
            new_people = set()
            for person in people:
                if not match_list_vec(people, person):
                    removed_people.add(person)
                else:
                    new_people.add(person)
            new_peoples.append(new_people)

    for person in removed_people:
        new_peoples.append({person})

    return new_peoples


print("1st step")
start = time.time()
#peoples_1st = make_1st_match(file_paths[:1000])
peoples_1st = make_1st_match(file_paths)
print("  + 1st step took {} seconds.".format(time.time() - start))

print("2nd step")
start = time.time()
match_rate = 0.0
peoples_2nd = merge_people(peoples_1st, match_rate)
print("  + 2nd step took {} seconds.".format(time.time() - start))
match_rate += 0.01

output_peoples = peoples_2nd
trial_count = 100
for i in range(trial_count):
    if i < trial_count // 2:
        print("3rd step {}/{}".format(i, trial_count))
        start = time.time()
        output_peoples = remove_less_match_people(output_peoples)
        print("  + 3rd step took {} seconds.".format(time.time() - start))

    print("4th step {}/{}".format(i, trial_count))
    start = time.time()
    before_size = len(output_peoples)
    output_peoples = merge_people(output_peoples, match_rate)
    print("merged {} => {}".format(before_size, len(output_peoples)))
    #print("  + 4th step took {} seconds.".format(time.time() - start))
    match_rate = min(match_rate + 0.01, 0.03)

## 書き出し
output_peoples = sorted_peoples(output_peoples)
with open("face_list.txt", mode='w') as f:
    for people in output_peoples:
        if len(people) >= 3:
            f.write("{}\n".format(",".join(people_to_short_path(people))))
print("congrats!")
