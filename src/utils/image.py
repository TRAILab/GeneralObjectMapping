import cv2


def resize_image(im, s):
    # Calculate the new dimensions
    new_dimensions = (int(im.shape[1] * s), int(im.shape[0] * s))

    # Resize the image
    resized_image = cv2.resize(im, new_dimensions, interpolation=cv2.INTER_AREA)

    return resized_image
