import cv2
from matplotlib import pyplot as plt
from main import get_ROI, mask_image


def make_diff_6_channels(room: str, id: int):
    ref = cv2.imread(f'../Images/{room}/Reference.JPG')
    image = cv2.imread(f'../Images/{room}/IMG_{id}.JPG')

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

    # plot the 6 channels on grid of 2x3
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(diff_L, cmap='gray')
    plt.title('L')
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(diff_A, cmap='gray')
    plt.title('A')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(diff_B, cmap='gray')
    plt.title('B')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(diff_H, cmap='gray')
    plt.title('H')
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(diff_S, cmap='gray')
    plt.title('S')
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(diff_V, cmap='gray')
    plt.title('V')
    plt.axis('off')

    plt.savefig(f'{room}/diff_{id}.png')


if __name__ == '__main__':
    for id in range(6551, 6561):
        make_diff_6_channels('Salon', id)
