import streamlit as st
import mediapipe as mp
import numpy as np
from tensorflow import keras
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.expected_num_features = model.n_features_in_

    def transform(self, frame):
        data_aux = []
        x_ = []
        y_ = []

        flipped_frame = cv2.flip(frame, 1)

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

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) < self.expected_num_features:
                data_aux.extend([0] * (self.expected_num_features - len(data_aux)))
            elif len(data_aux) > self.expected_num_features:
                data_aux = data_aux[:self.expected_num_features]

            prediction = model.predict([np.asarray(data_aux)])

            print("Predicted Value:", prediction[0])

            cv2.rectangle(flipped_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(flipped_frame, prediction[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        return flipped_frame


def main():
    st.title('Sign Language Recognition App with MediaPipe')

    app_mode = st.sidebar.selectbox('Select Mode', ['About App', 'Run in Real Time'])

    if app_mode == 'About App':
        # Add information about the app
        pass

    elif app_mode == 'Run in Real Time':
        st.write('Perform the sign language gesture in front of your webcam...')
        webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            mode=0,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            ),
        )


if __name__ == "__main__":
    main()
