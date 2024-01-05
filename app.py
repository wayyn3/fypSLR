import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

flipped_frame = None
x_ = []
y_ = []


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global flipped_frame, x_, y_
        img = frame.to_ndarray(format="bgr24")
        flipped_frame = cv2.flip(img, 1)
        H, W, _ = flipped_frame.shape
        frame_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    flipped_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            x_, y_ = [], []
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            expected_num_features = model.n_features_in_
            if len(data_aux) < expected_num_features:
                data_aux.extend([0] * (expected_num_features - len(data_aux)))
            elif len(data_aux) > expected_num_features:
                data_aux = data_aux[:expected_num_features]

            prediction = model.predict([np.asarray(data_aux)])

            cv2.rectangle(flipped_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(flipped_frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(flipped_frame, format="bgr24")


def sign_language_recognition_on_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    landmarks_list = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                landmarks_list.append(x - min(x_))
                landmarks_list.append(y - min(y_))

        expected_num_features = model.n_features_in_

        landmarks_list = np.asarray(landmarks_list)
        if len(landmarks_list) < expected_num_features:
            landmarks_list = np.concatenate([landmarks_list, np.zeros(expected_num_features - len(landmarks_list))])
        elif len(landmarks_list) > expected_num_features:
            landmarks_list = landmarks_list[:expected_num_features]

        prediction = model.predict([np.asarray(landmarks_list)])

        st.image(image, caption=f'Predicted Gesture: {prediction[0]}', use_column_width=True)


st.title('Sign Language Recognition App with MediaPipe')

app_mode = st.sidebar.selectbox('Select Mode', ['About App', 'Run with Image', 'Run in Real Time'])

if app_mode == 'About App':
    st.markdown(
        'This app performs real-time ASL Alphabet sign language recognition using a pre-trained machine learning model with MediaPipe hand detection.')
    st.markdown(
        'The app is divided into two modes: "Run in Real Time" and "Run with Image." In the "Run in Real Time" mode, the app initializes the webcam and utilizes the MediaPipe Hands module to detect hand landmarks in each frame. The detected landmarks are then processed to obtain the required features, and the pre-trained Random Forest model is used to predict the corresponding sign language gesture. The "Run with Image" mode allows users to upload an image for sign language recognition. The uploaded image is processed using the same pipeline as in the real-time mode, and the predicted gesture is displayed alongside the original image.')
    st.markdown(
        'The dataset is taken from Kaggle: https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet')
    st.header('Video Demo - Run In Real Time')
    local_video_path = './RealTimeDemo.mp4'
    st.video(local_video_path)

    st.header('Video Demo - Run with Image')
    local_video_path = './ImageDemo.mp4'
    st.video(local_video_path)

elif app_mode == 'Run with Image':
    image_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file_buffer:
        image = cv2.imdecode(np.frombuffer(image_file_buffer.read(), np.uint8), 1)
        sign_language_recognition_on_image(image)
        
elif app_mode == 'Run in Real Time':
    st.write('Perform the sign language gesture in front of your webcam...')

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor
    )
