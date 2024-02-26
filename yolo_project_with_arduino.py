import cv2
import numpy as np
import serial
import time

### 아두이노와 시리얼 통신 설정
ser = serial.Serial('/dev/ttyACM0', 9600)  # 시리얼 포트와 보드레이트를 맞춰주세요.

### yolo 설정
config_path = "/home/pi/darknet/cfg/yolov3.cfg"
weight_path = "/home/pi/darknet/yolov3.weights"
class_names = []
with open("/home/pi/darknet/data/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weight_path, config_path)

### 웹캠 연결
cap = cv2.VideoCapture(0)

### 웹캠 속성 설정
print('width: {0}, height: {1}'.format(cap.get(3), cap.get(4)))
cap.set(3, 320)
cap.set(4, 240)

person_detected = False
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ### yolo 입력 이미지 준비
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    ### 감지된 객체 정보 저장
    class_ids = []
    confidences = []
    boxes = []

    ### 감지된 객체 정보 분석
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 클래스가 사람일 경우에만
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    ### 비최대억제 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    ### 감지된 객체 그리기
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            person_detected = True

    if person_detected:
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time >= 10:  # 10초 동안 사람이 감지되면
            ser.write(b'1')  # 아두이노로 1을 보내서 스피커를 작동시킴
            print("Speaker activated!")
            start_time = None
            person_detected = False
    else:
        start_time = None

    ### 화면에 출력
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
