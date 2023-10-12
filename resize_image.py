import glob
import os
import cv2

if __name__ == '__main__':
    dst_dir = "image_small/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for p in sorted(list(glob.glob("image_original/*.jpg", recursive=False))):
        bgrImg = cv2.imread(p)
        dst = cv2.resize(bgrImg, None, None, 0.2, 0.2)
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(p)), dst)

