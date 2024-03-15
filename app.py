import math
import time

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)
end = 0
while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    fps = math.ceil(1 / (time.time()-end))
    end = start

    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    image_height, image_width, _ = frame_rgb.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for ids, landmrk in enumerate(hand_landmarks.landmark):
                # print(ids, landmrk)
                cx, cy = landmrk.x * image_width, landmrk.y * image_height
                print (ids, cx, cy)
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2),connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))

    # Display the frame with hand landmarks
    cv2.putText(frame,str(fps),(10,30),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),2 )
    cv2.imshow('Hand Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()