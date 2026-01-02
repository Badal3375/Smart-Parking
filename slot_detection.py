import cv2
import numpy as np


def detect_parking_slots(frame):
    """
    Automatically detect parking slots using contours
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    slots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter valid parking slot sizes
        if 50 < w < 180 and 80 < h < 250:
            slots.append((x, y, w, h))

    return slots


def check_occupancy(frame, slots):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 16
    )

    free = 0

    for x, y, w, h in slots:
        roi = thresh[y:y+h, x:x+w]
        count = cv2.countNonZero(roi)

        if count < 900:
            color = (0, 255, 0)  # Free
            free += 1
        else:
            color = (0, 0, 255)  # Occupied

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame, free
