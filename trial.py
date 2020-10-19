# https://openvino.jp/person-detection-raspberrypi/

import cv2 as cv

net = cv.dnn_DetectionModel('person-detection-retail-0013.xml',
                            'person-detection-retail-0013.bin')
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

video = cv.VideoCapture(0)
while True:
  # frame = cv.imread('/path/to/image')
  ret, frame = video.read()
  if frame is None:
    raise Exception('Image not found!')
  _, confidences, boxes = net.detect(frame, confThreshold=0.5)
  for confidence, box in zip(list(confidences), boxes):
    cv.rectangle(frame, box, color=(0, 255, 0))
    cv.putText(frame, str(confidence[0]), (box[0], box[0] + 15), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
  # cv.imwrite('out.png', frame)
  cv.imshow('title', frame)
  cv.waitKey(1)
