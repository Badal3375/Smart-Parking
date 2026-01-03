import cv2
import numpy as np
import mediapipe as mp
import math
import streamlit as st  # For Streamlit deployment

# --- Configuration ---
draw_color = (255, 0, 255)  # Default: Magenta (BGR)
brush_size = 15
eraser_size = 50

# --- MediaPipe Setup (Legacy API - works with mediapipe==0.10.14) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5, 
    max_num_hands=1
)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Canvas Setup ---
img_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
xp, yp = 0, 0

print("Air Canvas Pro Loaded: RED, GREEN, BLUE, YELLOW, ERASER")

# --- Instructions ---
print("Controls:")
print("- Index + Middle finger UP in header = Select color (Red/Green/Blue/Yellow/Eraser)")
print("- Index finger ONLY = Draw")
print("- All 5 fingers UP = Clear Canvas")
print("- Pinch (Thumb + Index) = Resize brush")
print("- Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    img = cv2.flip(img, 1)  # Mirror the image

    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # --- UI Header (Colors in BGR) ---
    # Header Background
    cv2.rectangle(img, (0, 0), (1280, 100), (50, 50, 50), cv2.FILLED)
    
    # Color buttons with labels
    buttons = [
        ((40, 10), (240, 90), (0, 0, 255), "RED"),
        ((290, 10), (490, 90), (0, 255, 0), "GREEN"),
        ((540, 10), (740, 90), (255, 0, 0), "BLUE"),
        ((790, 10), (990, 90), (0, 255, 255), "YELLOW"),
        ((1040, 10), (1240, 90), (0, 0, 0), "ERASER")
    ]
    
    for (x1, y1), (x2, y2), color, label in buttons:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 20, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = img.shape
            
            # Key landmarks (pixels)
            x1, y1 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)   # Index tip
            x2, y2 = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h) # Middle tip
            x_thumb, y_thumb = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)  # Thumb tip

            # Finger detection
            fingers = []
            tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            
            # Thumb (special case - horizontal)
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other fingers (vertical)
            for id in range(1, 5):
                if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id]-2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # --- CLEAR CANVAS (All 5 fingers up) ---
            if fingers == [1, 1, 1, 1, 1]:
                img_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(img, "CANVAS CLEARED!", (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

            # --- SELECTION MODE (Index + Middle up) ---
            elif fingers[1] == 1 and fingers[2] == 1:
                xp, yp = 0, 0  # Reset drawing position
                
                # Header selection (y1 < 100)
                if y1 < 100:
                    for (bx1, by1), (bx2, by2), color, _ in buttons:
                        if bx1 < x1 < bx2:
                            draw_color = color
                            break
                
                # Pinch-to-resize (Thumb + Index distance)
                length = math.hypot(x1 - x_thumb, y1 - y_thumb)
                brush_size = int(np.interp(length, [20, 150], [5, 50]))
                
                # Visual feedback
                cv2.circle(img, (x1, y1), brush_size, draw_color, 3)
                cv2.circle(img, (x1, y1), 10, draw_color, cv2.FILLED)
                
                if draw_color == (0, 0, 0):  # Eraser mode
                    eraser_size = brush_size
                cv2.putText(img, f"Brush: {brush_size}px", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

            # --- DRAWING MODE (Index up, Middle down) ---
            elif fingers[1] == 1 and fingers[2] == 0:
                current_size = eraser_size if draw_color == (0, 0, 0) else brush_size
                
                cv2.circle(img, (x1, y1), int(current_size), draw_color, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                # Draw thick line on canvas
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, current_size)
                xp, yp = x1, y1

            # Draw hand landmarks (optional visualization)
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- Canvas Overlay (Alpha blend) ---
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    
    img_bg = cv2.bitwise_and(img, img, mask=mask)
    img_canvas_fg = cv2.bitwise_and(img_canvas, img_canvas, mask=mask_inv)
    img = cv2.add(img_bg, img_canvas_fg)

    # Display
    cv2.imshow("Air Canvas Pro - Hand Drawing", img)
    
    # Instructions overlay
    cv2.putText(img, "'q' to quit", (1050, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Air Canvas closed.")
