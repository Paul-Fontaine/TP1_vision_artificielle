import cv2
import img2pdf
from PIL import Image
import io


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