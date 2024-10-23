import cv2
import numpy as np
import img2pdf
from PIL import Image
import io


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


def save_outputs2pdf(image_list, pdf_path):
    # Create a list to store the image bytes
    image_bytes_list = []

    for img in image_list:
        # Convert OpenCV image (BGR) to RGB for Pillow compatibility
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # Save the PIL image to a bytes buffer
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Add the image bytes to the list
        image_bytes_list.append(img_byte_arr)

    # Write images to PDF using img2pdf
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_bytes_list))


def detect_objects(room: str, id: int, threshold_value: int = 80, show_steps: bool = False, save_steps: bool = False):
    ref = cv2.imread(f'Images/{room}/Reference.JPG')
    image = cv2.imread(f'Images/{room}/IMG_{id}.JPG')
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
        cv2.imshow(f'diff_{id}', cv2.resize(diff, (0, 0), fx=0.15, fy=0.15))

    _, diff_thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    if show_steps:
        cv2.imshow(f'diff_thresh_{id}', cv2.resize(diff_thresh, (0, 0), fx=0.15, fy=0.15))

    kernel_morph = np.ones((25, 25), np.uint8)
    diff_morph = diff_thresh.copy()
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_CLOSE, kernel_morph)
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_OPEN, kernel_morph)
    diff_morph = cv2.dilate(diff_morph, kernel_morph, iterations=2)
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_CLOSE, kernel_morph)

    if show_steps:
        cv2.imshow(f'morph_{id}', cv2.resize(diff_morph, (0, 0), fx=0.15, fy=0.15))

    contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]

    image_contours = cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 4)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_contours, (x, y), (x + w, y + h), (0, 0, 255), 5)

    image_contours = cv2.polylines(image_contours, [ROI], True, (255, 0, 0), 3)

    if show_steps:
        cv2.imshow(f'image_contours_{id}', cv2.resize(image_contours, (0, 0), fx=0.15, fy=0.15))
        cv2.waitKey(0)

    if save_steps:
        save_outputs2pdf([ref, image, diff, diff_thresh, diff_morph, image_contours], f'Outputs/{room}/steps_{id}.pdf')

    return image_contours


if __name__ == '__main__':
    outputs = []

    ids4room = {'Chambre': range(6567, 6574), 'Cuisine': range(6562, 6566), 'Salon': range(6551, 6561)}
    for room in ['Chambre', 'Cuisine', 'Salon']:
        for id in ids4room[room]:
            output = detect_objects(room, id, threshold_value=80, save_steps=True)
            outputs.append(output)

        save_outputs2pdf(outputs, f'Outputs/{room}.pdf')
