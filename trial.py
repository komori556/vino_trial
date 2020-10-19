# https://openvino.jp/person-detection-raspberrypi/

import cv2 as cv
import fire


def detect_try(model_name: str = "person-detection-0106"):
  print(model_name + '.xml')
  net = cv.dnn_DetectionModel(model_name + '.xml', model_name + '.bin')
  net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

  video = cv.VideoCapture(0)
  while True:
    ret, frame = video.read()
    if frame is None:
      raise Exception('Image not found!')
    _, confidences, boxes = net.detect(frame, confThreshold=0.3)
    for confidence, box in zip(list(confidences), boxes):
      cv.rectangle(frame, box, color=(0, 255, 0))
      cv.putText(
        frame, str(confidence[0]), (box[0], box[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
        1)
    cv.imshow('title', frame)
    cv.waitKey(1)


if __name__ == '__main__':
  fire.Fire(detect_try)
