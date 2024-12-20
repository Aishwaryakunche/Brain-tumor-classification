import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import imutils
import cv2
import PIL.Image
import os

# Augmentation settings
demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)


def cropAndAugmentation():
    IMG_SIZE = 224
    dim = (IMG_SIZE, IMG_SIZE)
    directory = ['/home/aishwaryakunche/DL project/Testing/glioma_tumor']
    directory_output = ['/home/aishwaryakunche/DL project/testing/glioma_tumor_test']

    for input_folder in directory:
        for output_folder in directory_output:
            for img in os.listdir(input_folder):
                image = cv2.imread(os.path.join(input_folder, img))

                # Resize image
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                # Thresholding and finding contours
                thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=2)

                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                c = max(cnts, key=cv2.contourArea)

                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])

                ADD_PIXELS = 0
                # Cropping the image based on extreme points
                new_image = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

                x = new_image
                x = x.reshape((1,) + x.shape)

                # Generate 20 augmented images per input image
                for i, batch in enumerate(demo_datagen.flow(x, batch_size=1, save_to_dir=output_folder, 
                                                            save_prefix=os.path.splitext(img)[0], save_format='jpg')):
                    if i >= 20:  # Generate only 20 augmented images
                        break

                print(f"Processed {img} -> Generated 20 augmentations.")

cropAndAugmentation()

