import os
import dlib
import numpy as np
np.set_printoptions(precision=2)

def make_file_paths():
    file_paths_base = []
    for root, _dirs, files in os.walk("vecs"):
        for file in files:
            file_paths_base.append(os.path.join(root, file))
    return sorted(file_paths_base)

def load_vecs(file_paths):
    vecs = []
    for p in file_paths:
        v = np.load(p)
        vecs.append(dlib.vector(v))
    return vecs

def sorted_peoples(peoples):
    return sorted(peoples, reverse=True, key=len)

def get_path(p):
    s = p.split('/')
    return "/".join([s[len(s) - 2], s[len(s) - 1]])

def people_to_short_path(people):
    new_people = set()
    for person in people:
        p = get_path(person)
        new_people.add(p.replace(".npy", ""))
    return new_people

file_paths = make_file_paths()
vecs = load_vecs(file_paths)

labels = dlib.chinese_whispers_clustering(vecs, 0.3)
num_classes = len(set(labels))
peoples = []
for i in range(0, num_classes):
    indices = []
    for k, label in enumerate(labels):
        if label == i:
            indices.append(k)

    people = set()
    for index in indices:
        people.add(file_paths[index])
    peoples.append(people)

with open("face_list.txt", mode='w') as f:
    index = 0
    for people in sorted_peoples(peoples):
        if len(people) >= 3:
            f.write("{}\n".format(",".join(people_to_short_path(people))))
            print("{:04} {}".format(index, len(people)))
            index += 1
print("congrats!")
