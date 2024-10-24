import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

# Hand detection and drawing utilities from MediaPipe
# Limit detection to one hand for efficiency
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Mouse movement smoothing
smoothening = 5
plocx, plocy = 0, 0

# FPS variables
prev_frame_time = 0
new_frame_time = 0

# Variables to avoid repeated clicking
clicking_threshold = 70
clicked = False

while True:
    start_time = cv2.getTickCount()  # Record start time for FPS control

    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural hand movement
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB for hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            # Draw hand landmarks
            drawing_utils.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # Get coordinates of index finger tip (ID 8) and thumb tip (ID 4)
            index_x, index_y = int(
                landmarks[8].x * frame_width), int(landmarks[8].y * frame_height)
            thumb_x, thumb_y = int(
                landmarks[4].x * frame_width), int(landmarks[4].y * frame_height)

            # Visualize index finger tip
            cv2.circle(frame, (index_x, index_y), 15, (0, 255, 255), -1)

            # Map the hand coordinates to the screen
            screen_index_x = (screen_width / frame_width) * index_x
            screen_index_y = (screen_height / frame_height) * index_y

            # Smooth mouse movement
            clocx = plocx + (screen_index_x - plocx) / smoothening
            clocy = plocy + (screen_index_y - plocy) / smoothening
            pyautogui.moveTo(clocx, clocy)
            plocx, plocy = clocx, clocy

            # Check for click gesture (when index and thumb are close)
            cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 255, 255), -1)
            if abs(screen_index_y - (screen_height / frame_height) * thumb_y) < clicking_threshold:
                if not clicked:  # Only click once when in range
                    pyautogui.click()
                    clicked = True
            else:
                clicked = False  # Reset click flag when fingers move apart

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display FPS on frame
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Virtual Mouse with FPS', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
