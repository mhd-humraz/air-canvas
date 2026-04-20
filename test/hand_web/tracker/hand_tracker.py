import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.results = None  # store last results

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, handNo=0, draw=True):
        lm_list = []

        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

        return lm_list

    def fingers_up(self):
        fingers = []

        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]

            tips = [4, 8, 12, 16, 20]

            # Thumb (x-axis check)
            if hand.landmark[tips[0]].x < hand.landmark[tips[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers (y-axis check)
            for id in range(1, 5):
                if hand.landmark[tips[id]].y < hand.landmark[tips[id] - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
        else:
            fingers = [0, 0, 0, 0, 0]

        return fingers