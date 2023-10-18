import csv
import cv2
import os
import shutil
import numpy as np

def concat_tile(images_2d):
    return cv2.vconcat([cv2.hconcat(images_h) for images_h in images_2d])

rows = []
with open('face_list.txt', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

outputDir = "visualize"
shutil.rmtree(outputDir)

for index, row in enumerate(rows):
    images = []
    images_line = []
    for j, p in enumerate(row):
        imgPath = os.path.join("faces", p)
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        images_line.append(bgrImg)
        if len(images_line) >= 6:
            images.append(images_line)
            images_line = []
        if j >= (6 * 6) - 1:
            break
    if len(images_line) > 0:
        for i in range(6 - len(images_line)):
            images_line.append(np.zeros((96, 96, 3), np.uint8))
        images.append(images_line)
        images_line = []

    #print("{} {}".format(len(images), len(images[-1])))
    output_image = concat_tile(images)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    cv2.imwrite("{}/{:04}.jpg".format(outputDir, index), output_image)
