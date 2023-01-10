from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import os
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
        self.detectFPS = []
        self.detectLabels = []
        self.detectIoUs = []
        self.detectDices = []
        self.totalTime = 0.0
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
        count = 0
        for dir in os.listdir(folder_path):
            if  dir == 'image':
                for image in os.listdir(os.path.join(folder_path, dir)):
                    if image.split('.')[-1] == 'jpg' or image.split('.')[-1] == 'png':
                        img_path = os.path.join(folder_path, dir, image)
                        self.images.append(img_path)
                        count += 1
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
                    if confidence > .6:
                        center_x = int(detection[0] * wt)
                        center_y = int(detection[1] * ht)
                        w = int(detection[2] * wt)
                        h = int(detection[3] * ht)
                        
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .6, .5)
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
                    self.detectLabels.append(label)
            else:
                self.detectLabels.append("")
            
        self.isDetect = True
        loop_time = cv2.getTickCount() - loop_start
        total_time = loop_time / (cv2.getTickFrequency())
        self.detectFPS = len(self.images) / total_time
        self.ui.label_11.setText(str(self.detectFPS))
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
        self.ui.label_13.setText(self.detectLabels[num])