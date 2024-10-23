import cv2
import numpy as np
import sys


def get_ROI(room: str):
    if room == 'Chambre':
        ROI = np.array([
            [3176, 792],
            [5544, 2172],
            [3584, 3912],
            [1287, 2136]
        ], np.int32)
    elif room == 'Cuisine':
        ROI = np.array([
            [1100, 3999],
            [2000, 2000],
            [3600, 2000],
            [4500, 3999]
        ], np.int32)
    elif room == 'Salon':
        ROI = np.array([
            [0, 3999],
            [5900, 3999],
            [3290, 2250],
            [264, 2893]
        ], np.int32)
    else:
        raise ValueError('@param room must be one of the following: Chambre, Cuisine, Salon')

    return ROI


def mask_image(image, ROI):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [ROI], (255, 255, 255))
    return cv2.bitwise_and(image, mask)

def detect_objects(ref_path: str, image_path: str, room: str, threshold_value: int = 80, show_steps: bool = False, save_steps: bool = False):
    ref = cv2.imread(ref_path)
    image = cv2.imread(image_path)
    image_contours = image.copy()

    ROI = get_ROI(room)
    ref = mask_image(ref, ROI)
    image = mask_image(image, ROI)

    ref_LAB = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    ref_HSV = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ref_L, ref_A, ref_B = cv2.split(ref_LAB)
    ref_H, ref_S, ref_V = cv2.split(ref_HSV)
    image_L, image_A, image_B = cv2.split(image_LAB)
    image_H, image_S, image_V = cv2.split(image_HSV)

    diff_L = cv2.absdiff(ref_L, image_L)
    diff_A = cv2.absdiff(ref_A, image_A)
    diff_B = cv2.absdiff(ref_B, image_B)
    diff_H = cv2.absdiff(ref_H, image_H)
    diff_S = cv2.absdiff(ref_S, image_S)
    diff_V = cv2.absdiff(ref_V, image_V)

    diff = diff_L * 0.1 + diff_A * 0.3 + diff_B * 0.3 + diff_H * 0.4 + diff_S * 0.2
    if room == 'Salon':
        diff = diff * 0.3 + diff_V * 0.3 + diff_S * 0.2 + diff_A * 0.1 + diff_B * 0.1
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    diff = cv2.blur(diff, (25, 25))

    if show_steps:
        cv2.imshow(f'diff', cv2.resize(diff, (0, 0), fx=0.15, fy=0.15))

    _, diff_thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    if show_steps:
        cv2.imshow(f'diff_thresh', cv2.resize(diff_thresh, (0, 0), fx=0.15, fy=0.15))

    kernel_morph = np.ones((25, 25), np.uint8)
    diff_morph = diff_thresh.copy()
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_CLOSE, kernel_morph)
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_OPEN, kernel_morph)
    diff_morph = cv2.dilate(diff_morph, kernel_morph, iterations=2)
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_CLOSE, kernel_morph)

    if show_steps:
        cv2.imshow(f'morph', cv2.resize(diff_morph, (0, 0), fx=0.15, fy=0.15))

    contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]

    image_contours = cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 4)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_contours, (x, y), (x + w, y + h), (0, 0, 255), 5)

    image_contours = cv2.polylines(image_contours, [ROI], True, (255, 0, 0), 3)

    cv2.imshow(f'contours_{image_path}', cv2.resize(image_contours, (0, 0), fx=0.15, fy=0.15))
    cv2.waitKey(0)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python detect_objects.py <path/reference> <path/image> <room> [show_steps : '0' or '1'] optional")
        sys.exit(1)

    ref_path = sys.argv[1]
    image_path = sys.argv[2]
    room = sys.argv[3]
    if room not in ['Chambre', 'Cuisine', 'Salon']:
        print("Usage: python detect_objects.py <path/reference> <path/image> <room: 'Chambre' or 'Cuisine' or 'Salon'> [show_steps : '0' or '1'] optional")
        sys.exit(1)

    if len(sys.argv) == 5:
        if sys.argv[4] not in ['0', '1']:
            print("Usage: python detect_objects.py <path/reference> <path/image> <room> [show_steps : '0' or '1'] optional")
            sys.exit(1)
        show_steps = bool(sys.argv[4] )

    try:
        print(f"ref_path: {ref_path},\nimage_path: {image_path},\nroom: {room}")
        if show_steps:
            print(f"show_steps: {show_steps}")
        detect_objects(ref_path, image_path, room, show_steps=show_steps)
    except Exception as e:
        print("Usage: python detect_objects.py <path/reference> <path/image> <room: 'Chambre' or 'Cuisine' or 'Salon'> [show_steps : '0' or '1'] optional")
        print(e)
        sys.exit(1)
