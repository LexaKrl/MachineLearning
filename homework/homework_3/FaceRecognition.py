import cv2
import face_recognition
import mediapipe as mp
from deepface import DeepFace
import time

# === Настройки владельца ===
OWNER_NAME = "Aleksei"
OWNER_SURNAME = "Kirilin"
owner_image_path = 'owner.jpg'

# === Загрузка и кодирование фото владельца ===
owner_image = face_recognition.load_image_file(owner_image_path)
owner_encoding = face_recognition.face_encodings(owner_image)[0]

# === Инициализация MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Переменные ===
face_locations = []
face_encodings = []
owner_verified = False
label = "unknown"

last_face_check = 0
last_emotion_check = 0
FACE_CHECK_INTERVAL = 1.0

# === Подсчёт пальцев ===
def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# === Камера ===
cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

        # === Распознавание лица раз в секунду ===
        if time.time() - last_face_check > FACE_CHECK_INTERVAL:
            last_face_check = time.time()
            face_locations = face_recognition.face_locations(small_rgb)
            face_encodings = face_recognition.face_encodings(small_rgb, face_locations)
            owner_verified = False

            for encoding in face_encodings:
                matches = face_recognition.compare_faces([owner_encoding], encoding)
                if matches[0]:
                    owner_verified = True
                    break

        label = "unknown"
        if face_locations:
            # Увеличим координаты обратно до оригинального масштаба
            top, right, bottom, left = [int(pos * 4) for pos in face_locations[0]]
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            # === Обработка руки ===
            hand_results = hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                handLms = hand_results.multi_hand_landmarks[0]
                fingers = count_fingers(handLms)

                if fingers == 1 and owner_verified:
                    label = OWNER_NAME
                elif fingers == 2 and owner_verified:
                    label = OWNER_SURNAME
                elif fingers == 3 and owner_verified:
                    try:
                        face_crop = frame[top:bottom, left:right]
                        emotion = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False)
                        label = emotion[0]["dominant_emotion"]
                    except:
                        label = "emotion?"
                else:
                    label = "unknown"

                mp_drawing.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            else:
                label = "unknown"

            # === Надпись под лицом ===
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Face and Hand Detection", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
