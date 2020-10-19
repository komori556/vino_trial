# 下記のコピペ
# https://dev.classmethod.jp/articles/person-reidentification/
import numpy as np
import time
import random
import cv2
from openvino.inference_engine import IECore
from model import Model


class PersonDetector(Model):
  
  def __init__(self, model_path, device, ie_core, threshold, num_requests):
    super().__init__(model_path, device, ie_core, num_requests, None)
    _, _, h, w = self.input_size
    self.__input_height = h
    self.__input_width = w
    self.__threshold = threshold
  
  def __prepare_frame(self, frame):
    initial_h, initial_w = frame.shape[:2]
    scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
    in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape(self.input_size)
    return in_frame, scale_h, scale_w
  
  def infer(self, frame):
    in_frame, _, _ = self.__prepare_frame(frame)
    result = super().infer(in_frame)
    
    detections = []
    height, width = frame.shape[:2]
    for r in result[0][0]:
      conf = r[2]
      if (conf > self.__threshold):
        x1 = int(r[3] * width)
        y1 = int(r[4] * height)
        x2 = int(r[5] * width)
        y2 = int(r[6] * height)
        detections.append([x1, y1, x2, y2, conf])
    return detections


class PersonReidentification(Model):
  
  def __init__(self, model_path, device, ie_core, threshold, num_requests):
    super().__init__(model_path, device, ie_core, num_requests, None)
    _, _, h, w = self.input_size
    self.__input_height = h
    self.__input_width = w
    self.__threshold = threshold
  
  def __prepare_frame(self, frame):
    initial_h, initial_w = frame.shape[:2]
    scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
    in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
    in_frame = in_frame.transpose((2, 0, 1))
    in_frame = in_frame.reshape(self.input_size)
    return in_frame, scale_h, scale_w
  
  def infer(self, frame):
    in_frame, _, _ = self.__prepare_frame(frame)
    result = super().infer(in_frame)
    return np.delete(result, 1)


class Tracker:
  def __init__(self):
    # 識別情報のDB
    self.identifysDb = None
    # 中心位置のDB
    self.center = []
  
  def __getCenter(self, person):
    x = person[0] - person[2]
    y = person[1] - person[3]
    return (x, y)
  
  def __getDistance(self, person, index):
    (x1, y1) = self.center[index]
    (x2, y2) = self.__getCenter(person)
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    u = b - a
    return np.linalg.norm(u)
  
  def __isOverlap(self, persons, index):
    [x1, y1, x2, y2] = persons[index]
    for i, person in enumerate(persons):
      if (index == i):
        continue
      if (max(person[0], x1) <= min(person[2], x2) and max(person[1], y1) <= min(person[3], y2)):
        return True
    return False
  
  def getIds(self, identifys, persons):
    if (identifys.size == 0):
      return []
    if self.identifysDb is None:
      self.identifysDb = identifys
      for person in persons:
        self.center.append(self.__getCenter(person))
    
    print("input: {} DB:{}".format(len(identifys), len(self.identifysDb)))
    similaritys = self.__cos_similarity(identifys, self.identifysDb)
    similaritys[np.isnan(similaritys)] = 0
    ids = np.nanargmax(similaritys, axis=1)
    
    for i, similarity in enumerate(similaritys):
      persionId = ids[i]
      d = self.__getDistance(persons[i], persionId)
      print("persionId:{} {} distance:{}".format(persionId, similarity[persionId], d))
      # 0.95以上で、重なりの無い場合、識別情報を更新する
      if (similarity[persionId] > 0.95):
        if (self.__isOverlap(persons, i) == False):
          self.identifysDb[persionId] = identifys[i]
      # 0.5以下で、距離が離れている場合、新規に登録する
      elif (similarity[persionId] < 0.5):
        if (d > 500):
          print("distance:{} similarity:{}".format(d, similarity[persionId]))
          self.identifysDb = np.vstack((self.identifysDb, identifys[i]))
          self.center.append(self.__getCenter(persons[i]))
          ids[i] = len(self.identifysDb) - 1
          print("> append DB size:{}".format(len(self.identifysDb)))
    
    print(ids)
    # 重複がある場合は、信頼度の低い方を無効化する
    for i, a in enumerate(ids):
      for e, b in enumerate(ids):
        if (e == i):
          continue
        if (a == b):
          if (similarity[a] > similarity[b]):
            ids[i] = -1
          else:
            ids[e] = -1
    print(ids)
    return ids
  
  # コサイン類似度
  # 参考にさせて頂きました: https://github.com/kodamap/person_reidentification
  def __cos_similarity(self, X, Y):
    m = X.shape[0]
    Y = Y.T
    return np.dot(X, Y) / (
        np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0)
    )


# device = "CPU"
device = "MYRIAD"
cpu_extension = None
ie_core = IECore()
if device == "CPU" and cpu_extension:
  ie_core.add_extension(cpu_extension, "CPU")

THRESHOLD = 0.8
person_detector = PersonDetector("./person-detection-retail-0013", device, ie_core, THRESHOLD, num_requests=2)

personReidentification = PersonReidentification("./person-reidentification-retail-0079", device, ie_core, THRESHOLD,
                                                num_requests=2)
tracker = Tracker()

# MOVIE = "./video001.mp4"
# MOVIE = "./video002.mp4"
MOVIE = 0
SCALE = 0.3

cap = cv2.VideoCapture(MOVIE)

TRACKING_MAX = 50
colors = []
for i in range(TRACKING_MAX):
  b = random.randint(0, 255)
  g = random.randint(0, 255)
  r = random.randint(0, 255)
  colors.append((b, g, r))

while True:
  
  grabbed, frame = cap.read()
  if not grabbed:  # ループ再生
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    continue
  if (frame is None):
    continue
  
  # Personを検知する
  persons = []
  detections = person_detector.infer(frame)
  if (len(detections) > 0):
    print("-------------------")
    for detection in detections:
      x1 = int(detection[0])
      y1 = int(detection[1])
      x2 = int(detection[2])
      y2 = int(detection[3])
      conf = detection[4]
      print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
      persons.append([x1, y1, x2, y2])
  
  print("====================")
  # 各Personの画像から識別情報を取得する
  identifys = np.zeros((len(persons), 255))
  for i, person in enumerate(persons):
    # 各Personのimage取得
    img = frame[person[1]: person[3], person[0]: person[2]]
    h, w = img.shape[:2]
    if (h == 0 or w == 0):
      continue
    # identification取得
    identifys[i] = personReidentification.infer(img)
  
  # Idの取得
  ids = tracker.getIds(identifys, persons)
  
  # 枠及びIdを画像に追加
  for i, person in enumerate(persons):
    if (ids[i] != -1):
      color = colors[int(ids[i])]
      frame = cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), color, int(50 * SCALE))
      frame = cv2.putText(frame, str(ids[i]), (person[0], person[1]), cv2.FONT_HERSHEY_PLAIN, int(50 * SCALE), color,
                          int(30 * SCALE), cv2.LINE_AA)
  
  # 画像の縮小
  h, w = frame.shape[:2]
  frame = cv2.resize(frame, ((int(w * SCALE), int(h * SCALE))))
  # 画像の表示
  cv2.imshow('frame', frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
