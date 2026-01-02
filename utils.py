import cv2
import numpy as np

def check_parking_space(img, slots, threshold=900):
    free_spaces = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16
    )

    for x, y, w, h in slots:
        roi = thresh[y:y+h, x:x+w]
        count = cv2.countNonZero(roi)

        if count < threshold:
            color = (0, 255, 0)  # Free
            free_spaces += 1
        else:
            color = (0, 0, 255)  # Occupied

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    return img, free_spaces
