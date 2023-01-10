from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import os
import json
import cv2
from cv2 import getTickCount, getTickFrequency
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.isDetect = False
        self.isSegment = False
        self.images = []
        self.imgNames = []
        self.gtBoxes = []
        self.AP50 = []
        self.detectIoUs = []
        self.detectDices = []
        self.segmentedImages = []

    def setup_control(self):
        # TODO
        self.ui.pushButton.clicked.connect(self.readFolder)
        self.ui.pushButton_2.setEnabled(False)
        self.ui.pushButton_3.setEnabled(False)
        self.ui.pushButton_4.setEnabled(False)
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_2.clicked.connect(self.detectObject)
        self.ui.pushButton_3.clicked.connect(self.segmentImage)
        self.ui.pushButton_4.clicked.connect(self.lastImage)
        self.ui.pushButton_5.clicked.connect(self.nextImage)
    
    def readFolder(self):
        self.images = []
        folder_path = QFileDialog.getExistingDirectory(self,"Open folder","./")                 # start path
        self.ground_truth = folder_path.split('/')[-1]
        self.ui.label_12.setText(self.ground_truth)
        for dir in os.listdir(folder_path):
            if dir == 'image':
                for image in os.listdir(os.path.join(folder_path, dir)):
                    if image.split('.')[-1] == 'jpg' or image.split('.')[-1] == 'png':
                        img_path = os.path.join(folder_path, dir, image)
                        self.images.append(img_path)
            elif dir == 'label':
                for label in os.listdir(os.path.join(folder_path, dir)):
                    with open(os.path.join(folder_path, dir, label), 'r') as f:
                        data = json.load(f)
                        self.gtBoxes.append(data['shapes'])

        if len(self.images) == 1:
            self.ui.pushButton_5.setEnabled(False)
        else: 
            self.ui.pushButton_5.setEnabled(True)
        self.img = QPixmap(self.images[0]).scaled(self.ui.image_label.width(), self.ui.image_label.height())
        self.ui.image_label.setPixmap(self.img)
        self.ui.label_10.setText("1/" + str(len(self.images)))
        self.ui.pushButton_2.setEnabled(True)
        self.ui.pushButton_3.setEnabled(True)

    def lastImage(self):
        currentImg = int(self.ui.label_10.text().split('/')[0]) - 1
        nowImg = currentImg - 1     # index
        if nowImg == 0:
            self.ui.pushButton_4.setEnabled(False)
        # print(nowImg)
        self.ui.pushButton_5.setEnabled(True)
        self.img = QPixmap(self.images[nowImg]).scaled(self.ui.image_label.width(), self.ui.image_label.height())
        self.ui.image_label.setPixmap(self.img)
        self.ui.label_10.setText(str(nowImg + 1) + "/" + str(len(self.images)))
        if self.isDetect:
            self.showDetectInfo(nowImg)

    def nextImage(self):
        currentImg = int(self.ui.label_10.text().split('/')[0]) - 1
        nowImg = currentImg + 1
        if nowImg == len(self.images) - 1:
            self.ui.pushButton_5.setEnabled(False)
        # print(nowImg)
        self.ui.pushButton_4.setEnabled(True)
        self.img = QPixmap(self.images[nowImg]).scaled(self.ui.image_label.width(), self.ui.image_label.height())
        self.ui.image_label.setPixmap(self.img)
        self.ui.label_10.setText(str(nowImg + 1) + "/" + str(len(self.images)))
        if self.isDetect:
            self.showDetectInfo(nowImg)

    def detectObject(self):
        net = cv2.dnn.readNetFromDarknet("YOLO/yolov3_custom.cfg", "YOLO/yolov3_custom_last.weights")
        classes = ['powder_uncover', 'powder_uneven', 'scratch']
        loop_start = getTickCount()
        for index, img in enumerate(self.images):
            targetImg = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            targetImg = cv2.resize(targetImg,(1024, 1024))
            imgName = img.split('\\')[-1]
            self.imgNames.append(imgName)
            ht, wt, _ = targetImg.shape
            blob = cv2.dnn.blobFromImage(targetImg, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
            net.setInput(blob)
            last_layer = net.getUnconnectedOutLayersNames()
            layer_out = net.forward(last_layer)
            boxes = []
            confidences = []
            class_ids = []
            for output in layer_out:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]
                    if confidence > .3:
                        center_x = int(detection[0] * wt)
                        center_y = int(detection[1] * ht)
                        w = int(detection[2] * wt)
                        h = int(detection[3] * ht)
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .6, .3)
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x,y,w,h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(targetImg, (x, y), (x+w, y+h), color, 10)

                    detectedImg = im.fromarray(targetImg)
                    detectedImg.save('result/' + imgName)
                    self.detectLabel = label

            if len(boxes) != 0:
                maxIoU = [] # GT Box IoU
                TP = 0
                FP = 0
                for box in self.gtBoxes[index]:                                
                    gtBox = [box['points'][0][0], box['points'][0][1], box['points'][1][0], box['points'][1][1]]
                    if self.ground_truth == 'scratch':
                        iw = 1254
                        ih = 1244
                    else:
                        iw = 3384
                        ih = 3330
                    gtBox = self.resizeGT(gtBox, iw, ih)
                    IoU = [] # to find the max IoU prediction
                    for pBox in boxes:
                        predictBox = self.exchange(pBox)
                        result = self.getIoU(predictBox, gtBox)
                        IoU.append(result)
                    maxIoU.append(max(IoU))
                    if max(IoU) > .5:
                        TP += 1
                    else:
                        FP += 1
                self.detectIoUs.append(np.mean(maxIoU))
                precision = TP / (TP + FP)
                self.AP50.append(precision)
            else:
                self.detectIoUs.append(0.0)
                self.AP50.append(0.0)
            
        self.isDetect = True
        loop_time = cv2.getTickCount() - loop_start
        total_time = loop_time / (cv2.getTickFrequency())
        self.detectFPS = len(self.images) / total_time
        self.ui.label_11.setText(str(self.detectFPS))
        if self.ground_truth == 'powder_uncover':
            self.ui.label_21.setText(str(np.mean(self.AP50)))
        elif self.ground_truth == 'powder_uneven':
            self.ui.label_22.setText(str(np.mean(self.AP50)))
        elif self.ground_truth == 'scratch':
            self.ui.label_23.setText(str(np.mean(self.AP50)))
        self.showDetectInfo(0)

    def segmentImage(self):
        return
    
    def showDetectInfo(self, num):
        imgName = self.imgNames[num]
        if imgName not in os.listdir('result/'):
            self.detectedImg = self.img
        else:
            self.detectedImg = QPixmap('result/' + imgName).scaled(self.ui.image_label.width(), self.ui.image_label.height())
        self.ui.image_label_2.setPixmap(self.detectedImg)
        self.ui.label_14.setText(str(self.detectIoUs[num])) 
        if self.detectLabel != '':
            self.ui.label_13.setText(self.detectLabel)
    
    def resizeGT(self, box, iw, ih):
        box_resize = []
        scale_x = 1024 / iw
        scale_y = 1024 / ih
        nw = int(iw * scale_x)
        nh = int(ih * scale_y)
        dx = (1024 - nw) // 2
        dy = (1024 - nh) // 2
        box_resize.append(int(int(box[0]) * (1024 / iw) + dx))
        box_resize.append(int(int(box[1]) * (1024 / ih) + dy))
        box_resize.append(int(int(box[2]) * (1024 / iw) + dx))
        box_resize.append(int(int(box[3]) * (1024 / ih) + dy))
        return box_resize

    def exchange(self, box1):
        x,y = box1[0], box1[1]
        w,h = box1[2], box1[3]
        box2 = []
        box2.append(int(x))
        box2.append(int(y))
        box2.append(int(x+w))
        box2.append(int(y+h))
        return box2
    
    def getIoU(self, box1, box2):
        width1 = abs(box1[2] - box1[0])
        height1 = abs(box1[1] - box1[3])
        width2 = abs(box2[2] - box2[0])
        height2 = abs(box2[1] - box2[3])
        xmax = max(box1[0], box1[2], box2[0], box2[2])
        ymax = max(box1[1], box1[3], box2[1], box2[3])
        xmin = min(box1[0], box1[2], box2[0], box2[2])
        ymin = min(box1[1], box1[3], box2[1], box2[3])
        W = xmin + width1 + width2 - xmax
        H = ymin + height1 + height2 - ymax
        if W <= 0 or H <= 0:
            iou_ratio = 0
        else: 
            iou_area  = W * H
            box1_area = width1 * height1
            box2_area = width2 * height2
            iou_ratio = iou_area / (box1_area + box2_area - iou_area)
        return iou_ratio