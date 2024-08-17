from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import mediapipe as mp
import keyboard
import time
import unicode

# 한글을 지원하는 폰트 파일 경로
FONT_PATH = "C:\\Windows\\Fonts\\malgun.ttf"

def put_text(img, text, position, font_path, font_size, color):
    # 이미지에서 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)

    # 다시 OpenCV 이미지로 변환
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


max_num_hands = 10

gesture = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ',
    23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅘ', 29: 'ㅙ', 30: 'ㅚ', 31: 'ㅝ',
    32: 'ㅞ', 33: 'ㅟ', 34: 'ㅢ', 35: 'space_bar', 36: 'clear'
}
# 14: 'ㄲ', 15: 'ㄸ', 16: 'ㅃ', 17: 'ㅆ', 18: 'ㅉ',
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open 'test.txt' file with UTF-8 encoding
with open('test.txt', 'w', encoding='utf-8') as f:
    file = np.genfromtxt('dataSet.txt', delimiter=',')
    angleFile = file[:, :-1]
    labelFile = file[:, -1]
    angle = angleFile.astype(np.float32)
    label = labelFile.astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1
while True:
    ret, img = cap.read()
    if not ret:
        continue
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            comparev1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            comparev2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angle = np.arccos(np.einsum('nt,nt->n', comparev1, comparev2))

            angle = np.degrees(angle)
            if keyboard.is_pressed('a'):
                with open('test.txt', 'a', encoding='utf-8') as f:
                    for num in angle:
                        num = round(num, 6)
                        f.write(f'{num},')
                    f.write("27.000000\n")
                print("next")

            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            index = int(results[0][0])
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 35:
                            sentence += ' '
                        elif index == 36:
                            sentence = ''
                        else:
                            sentence += gesture[index]
                        startTime = time.time()

                # Use Pillow to draw text
                img = put_text(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10),
                                                            int(res.landmark[0].y * img.shape[0] + 40)),
                              FONT_PATH, 30, (255, 255, 255))

            # 자모 결합 후 화면에 출력
            joined_text = unicode.join_jamos(sentence)
            img = put_text(img, joined_text, (20, 410), FONT_PATH, 50, (255, 255, 255))

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('HandTracking', img)
    cv2.waitKey(100)
    if keyboard.is_pressed('b'):
        break

cap.release()
cv2.destroyAllWindows()