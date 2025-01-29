import mediapipe as mp
import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk


class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_finger_extended(self, landmarks, tip_index):
        """
        Determine if a specific finger is extended
        tip_index: landmark index of finger tip
        """
        tip_to_base = {
            8: 5,   # Index finger
            12: 9,  # Middle finger
            16: 13, # Ring finger
            20: 17, # Pinky
            4: 2    # Thumb
        }

        tip = landmarks[tip_index]
        base = landmarks[tip_to_base[tip_index]]
        wrist = landmarks[0]

        # Check vertical position and distance from wrist
        return (tip[1] < base[1] and
                self.calculate_distance(tip, wrist) >
                self.calculate_distance(base, wrist) * 1.1)

    def is_thumb_up(self, landmarks):
        """Detect Thumbs Up gesture"""
        thumb_tip = landmarks[4]

        # Check if thumb_tip is above all other landmarks
        return all(thumb_tip[1] < landmarks[i][1] for i in range(0, 21) if i != 4)

    def is_thumb_down(self, landmarks):
        """Detect Thumbs Down gesture"""
        thumb_tip = landmarks[4]

        # Check if thumb_tip is below all other landmarks
        return all(thumb_tip[1] > landmarks[i][1] for i in range(0, 21) if i != 4)

    def is_four_gesture(self, landmarks):
        """
        Detect Number 4 gesture: Thumb tucked, and index, middle, ring, and pinky fingers extended.
        """
        # Check if thumb points (0, 1, 2, 3, 4) are below the base of other fingers
        thumb_tucked = all(
            landmarks[i][1] > landmarks[j][1]
            for i in range(1, 5)  # Thumb points
            for j in range(5, 21)  # Other finger points
        )

        # Check if the four fingers are extended
        fingers_extended = (
            self.is_finger_extended(landmarks, 8) and  # Index
            self.is_finger_extended(landmarks, 12) and  # Middle
            self.is_finger_extended(landmarks, 16) and  # Ring
            self.is_finger_extended(landmarks, 20)  # Pinky
        )

        return thumb_tucked and fingers_extended

    def recognize_gesture(self, landmarks):
        """
        Comprehensive gesture recognition
        """
        landmarks = [(lm.x, lm.y) for lm in landmarks]

        finger_states = [
            self.is_finger_extended(landmarks, 8),
            self.is_finger_extended(landmarks, 12),
            self.is_finger_extended(landmarks, 16),
            self.is_finger_extended(landmarks, 20)
        ]

        if self.is_thumb_up(landmarks):
            return "Thumbs Up"
        elif self.is_thumb_down(landmarks):
            return "Thumbs Down"
        elif all(finger_states):
            return "Open Hand/5"
        elif self.is_four_gesture(landmarks):
            return "Number 4"
        elif all(not state for state in finger_states):
            return "Fist/0"
        elif finger_states[0] and not any(finger_states[1:]):
            return "Pointing/1"
        elif finger_states[0] and finger_states[1] and not any(finger_states[2:]):
            return "Peace/Victory/2"
        elif finger_states[0] and finger_states[1] and finger_states[2] and not finger_states[3]:
            return "Number 3"


        return "Unknown Gesture"

    def run_recognition(self):
        """
        Run hand gesture recognition using Tkinter
        """
        root = tk.Tk()
        root.title("Hand Gesture Recognition")
        root.geometry("640x580")

        # Video capture label
        label = tk.Label(root)
        label.pack()

        # Gesture display label
        gesture_label = tk.Label(
            root,
            text="No Gesture",
            font=("Arial", 20, "bold"),
            fg="blue"
        )
        gesture_label.pack(pady=10)

        # Capture video
        cap = cv2.VideoCapture(0)

        def process_frame():
            ret, frame = cap.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            current_gesture = "No Gesture"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    current_gesture = self.recognize_gesture(hand_landmarks.landmark)

                    for landmark in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            gesture_label.config(text=current_gesture)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(10, process_frame)

        process_frame()
        root.mainloop()
        cap.release()


if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.run_recognition()
