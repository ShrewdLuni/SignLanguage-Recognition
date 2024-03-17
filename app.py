import math
import pickle
import time

import cv2
import numpy
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
end = 0

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = {0: 'A', 1: 'B', 2: 'C'}

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    fps = math.ceil(1 / (time.time()-end))
    end = start

    data = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_height, image_width, _ = frame_rgb.shape

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                data.append(hand_landmarks.landmark[i].x)
                data.append(hand_landmarks.landmark[i].y)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=2),connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))

        prediction = labels_dict[int(model.predict([numpy.asarray(data)])[0])]

        print(prediction)


    cv2.putText(frame,str(fps),(10,30),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),2 )

    cv2.imshow('Hand Recognition', frame)
    cv2.waitKey(1)

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()