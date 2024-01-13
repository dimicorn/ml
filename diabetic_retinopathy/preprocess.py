import os
import numpy as np
import cv2
import sys

def scaleRadius(img: np.array, scale: int):
    x = img[img.shape[0]//2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)

def image_prep(new_path: str, old_path: str, f: str) -> np.array:
    scale = 300
    a = cv2.imread(f'{old_path}/{f}')
    a = scaleRadius(a, scale) # scale img to a given radius
    # subtract local mean color
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale / 30), -4, 128)
    b = np.zeros(a.shape) # remove outer 10%
    cv2.circle(b, (a.shape[1]//2, a.shape[0]//2), int(scale*0.9), (1, 1, 1), -1, 8, 0)
    a = a * b + 128 * (1-b)
    cv2.imwrite(f'{new_path}/{f}', a)

def main() -> None:
    try:
        old_path, new_path = sys.argv[1], sys.argv[2]
        images = [file for file in os.listdir(old_path) if file[-5:] == '.jpeg']
        for image in images:
            image_prep(new_path, old_path, image)
    except IndexError:
        print('Please input path/to/dataset & path/to/preprocessed/dataset as command line args')

if __name__ == '__main__':
    main()

