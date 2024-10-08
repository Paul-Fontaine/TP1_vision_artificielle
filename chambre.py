import cv2
import numpy as np


def detect_objects_chambre(id):
    piece = 'Chambre'
    ref = cv2.imread(f'Images/{piece}/Reference.JPG')
    image = cv2.imread(f'Images/{piece}/IMG_{id}.JPG')
    image_contours = image.copy()


    def get_mask():
        mask = np.zeros_like(image)
        if piece == 'Chambre':
            ROI = np.array([
                [3176, 792],
                [5544, 2172],
                [3584, 3912],
                [1287, 2136]
            ], np.int32)
        elif piece == 'Cuisine':
            ROI = np.array([
                [1100, 3999],
                [2000, 2000],
                [3600, 2000],
                [4500, 3999]
            ], np.int32)
        elif piece == 'Salon':
            ROI = np.array([
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]
            ], np.int32)
        else:
            raise ValueError('@param piece must be one of the following: Chambre, Cuisine, Salon')

        cv2.fillPoly(mask, [ROI], (255, 255, 255))
        return mask, ROI


    ref = cv2.bitwise_and(ref, get_mask()[0])
    image = cv2.bitwise_and(image, get_mask()[0])

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
    diff = cv2.convertScaleAbs(diff)

    T, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, diff_thresh = cv2.threshold(diff, int(T * 0.9), 255, cv2.THRESH_BINARY)

    kernel = np.ones((25, 25), np.uint8)
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=2)
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
    diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=1)
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    image_contours = cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 4)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_contours, (x, y), (x + w, y + h), (0, 0, 255), 5)

    image_contours = cv2.polylines(image_contours, [get_mask()[1]], True, (255, 0, 0), 3)

    cv2.imshow(f'image_contours_{id}', cv2.resize(image_contours, (0, 0), fx=0.2, fy=0.2))
    cv2.waitKey(0)


if __name__ == '__main__':
    for i in range(6562, 6566):
        detect_objects_chambre(i)
