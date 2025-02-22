import pickle

import cv2
import mediapipe as mp
import numpy as np

import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

start_time = time.time()

user_input = ""
old_prediction = ""

while True:
    data = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=2), connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data.append(x - min(x_))
                data.append(y - min(y_))

        x1 = int(min(x_) * w) - 10
        y1 = int(min(y_) * h) - 10

        x2 = int(max(x_) * w) - 10
        y2 = int(max(y_) * h) - 10
        if len(data) == 42:
            prediction = model.predict([np.asarray(data)])[0]
            if old_prediction != prediction:
                old_prediction = prediction
                start_time = time.time()
            elif time.time() - start_time > 1 and old_prediction == prediction:
                user_input = user_input + prediction
                start_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
        cv2.putText(frame, prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, user_input, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Hand Recognition', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
