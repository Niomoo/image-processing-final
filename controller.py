from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import os
import json
import glob, sys
import cv2
from cv2 import getTickCount, getTickFrequency
import numpy as np
from PIL import Image
from PIL import ImageFile
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image as im
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import albumentations as albu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import copy

from tqdm import tqdm

# # retina
import utils
import torchvision
from engine import train_one_epoch, evaluate
float_formatter = "{:.2f}".format

class conv3x3_bn_relu(nn.Module):
    def __init__(self, in_planes, kernels, stride, padding=1):
        super().__init__()

        self.sequence=nn.Sequential(
            nn.Conv2d(in_planes, kernels, kernel_size=3, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=kernels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequence(x)

class down(nn.Module): #max pool
    def __init__(self, in_planes, kernels, stride, padding=1,
          pooling=True):
        super().__init__()
        self.conv=nn.Sequential(
            # 3x3 convolution + BN(batch normalization) + relu
            conv3x3_bn_relu(in_planes, kernels, stride, padding),
            conv3x3_bn_relu(kernels, kernels, stride, padding),
        )
        self.max_pooling = nn.MaxPool2d(2) if pooling else nn.Identity()

    def forward(self, x):
        return self.conv(self.max_pooling(x))

class up(nn.Module):
    def __init__(self, in_planes, kernels, stride=1, padding=1,
          bilinear=True):
        super().__init__()
        if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = nn.Sequential(conv3x3_bn_relu(in_planes + in_planes // 2, kernels, stride, padding),
          conv3x3_bn_relu(kernels, kernels, stride, padding),
      )
        else:
                self.up = nn.ConvTranspose2d(in_planes, in_planes//2, kernels_size=2, stride=stride)
                self.conv = nn.Sequential(conv3x3_bn_relu(in_planes, kernels, stride, padding), conv3x3_bn_relu(kernels, kernels, stride, padding),
      )

    def forward(self, x1, x2):
        x2 = self.up(x2)
        #print(x2.shape)
        return self.conv(torch.cat([x1, x2], dim=1))

class UNet(nn.Module): # 1, 3, 1024, 1024
    def __init__(self, n_channel=1, n_class=1, bilinear=True):
        super().__init__()
        self.down1 = down(n_channel, 12, 1, pooling=False)
        self.down2 = down(12, 24, 1)
        self.down3 = down(24, 48, 1)
        self.down4 = down(48, 96, 1)
        self.down5 = down(96, 192, 1)
        self.down6 = down(192, 384, 1) # add a new layer
        self.up1 = up(192*2, 192, 1, bilinear=bilinear)
        self.up2 = up(96*2, 96, 1, bilinear=bilinear)
        self.up3 = up(48*2, 48, 1, bilinear=bilinear)
        self.up4 = up(24*2, 24, 1, bilinear=bilinear)
        self.up5 = up(12*2, 12, 1, bilinear=bilinear)
        self.out = nn.Conv2d(12, n_class, 3, 1, padding=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5) #add a new layer
        x = self.up1(x5, x6) + x5
        x = self.up2(x4, x) + x4
        x = self.up3(x3, x) + x3
        x = self.up4(x2, x) + x2
        x = self.up5(x1, x) + x1
        logit = self.out(x)
        return logit

class AugmentedImageDataset(Dataset):
    def __init__(self, images, labels, augmentation=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentation = augmentation
        
    def __getitem__(self, index):

        # Get path
        img = self.images[index]
        label = self.labels[index]
        
        # Read image
        img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        label = cv2.imdecode(np.fromfile(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # Augmentation
        if(self.augmentation):
            sampled = self.augmentation(image=img, mask=label)
            img = sampled['image']
            label = sampled['mask']
        
        # 增加channel這個維度
#         img = img[:,:,np.newaxis]
#         img = img.permute(1, 2, 0)
        label = label[np.newaxis,:,:]
        
        return torch.div(img, 255), torch.div(label, 255)

    def __len__(self):
        return len(self.images)

class Dice(nn.Module):
    def __init__(self, eps=1e-7, threshold=0.5):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.activation = torch.sigmoid

    @property
    def __name__(self):
        return 'Dice'

    def _threshold(self, x):
        return (x > self.threshold).type(x.dtype)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = self._threshold(y_pr)

        tp = torch.sum(y_gt * y_pr)
        fp = torch.sum(y_pr) - tp
        fn = torch.sum(y_gt) - tp

        score = torch.div((2 * tp + self.eps), (2 * tp + fn + fp + self.eps))
        return score

class AverageValueMeter(object):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 1:
            self.mean = 0.0 + self.sum
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0

        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.mean
        self.mean_old = np.nan
        self.m_s = 0.0
        self.std = np.nan

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
        self.detectLabel = ''
        self.masks = []
        self.segmentedImages = []
        self.segmentDices = []

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
            elif dir == 'mask':
                for image in os.listdir(os.path.join(folder_path, dir)):
                    if image.split('.')[-1] == 'jpg' or image.split('.')[-1] == 'png':
                        img_path = os.path.join(folder_path, dir, image)
                        self.masks.append(img_path)


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
        if self.isSegment:
            self.showSegmentInfo(nowImg)

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
        if self.isSegment:
            self.showSegmentInfo(nowImg)

    def detectObject(self):
        net = cv2.dnn.readNetFromDarknet("YOLO/yolov3_custom.cfg", "YOLO/yolov3_custom_last.weights")
        classes = ['powder_uncover', 'powder_uneven', 'scratch']
        loop_start = getTickCount()
        for index, img in enumerate(self.images):
            targetImg = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
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

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .3)
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x,y,w,h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(targetImg, (x, y), (x+w, y+h), color, 10)
                    
                    detectedImg = im.fromarray(targetImg) # numpy.ndarray to PIL.Image
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
                print(maxIoU)
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
    
    def showDetectInfo(self, num):
        imgName = self.imgNames[num]
        if imgName not in os.listdir('result/'):
            self.detectedImg = self.img
        else:
            self.detectedImg = QPixmap('result/' + imgName).scaled(self.ui.image_label_2.width(), self.ui.image_label_2.height())
        self.ui.image_label_2.setPixmap(self.detectedImg)
        self.ui.label_14.setText(str(self.detectIoUs[num])) 
        if self.detectLabel != '':
            self.ui.label_13.setText(self.detectLabel)
        else:
            self.ui.label_13.setText(self.ground_truth)
    
    def showSegmentInfo(self, num):
        imgName = self.imgNames[num]
        if imgName not in os.listdir('result/mask/'):
            self.segmentedImg = self.img
        else:
            self.segmentedImg = QPixmap('result/mask/' + imgName).scaled(self.ui.image_label_3.width(), self.ui.image_label_3.height())
        self.ui.image_label_3.setPixmap(self.segmentedImg)
        self.ui.label_15.setText(str(self.segmentDices[num]))

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

    def get_simple_training_augmentation():
        aug = [
            albu.GlassBlur(p=0.2),
            albu.GaussNoise(var_limit=(1,3),p=0.3), 
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, always_apply=False, p=0.2),
            albu.HorizontalFlip(p=0.5),  
            albu.Resize(height=1024, width=1024, p=1),
    #         albu.PadIfNeeded(1024, 1024, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2(transpose_mask=True)
        ]
        return albu.Compose(aug)


    def get_validation_augmentation(self):
        aug = [
            albu.Resize(height=1024, width=1024, p=1),
    #          albu.PadIfNeeded(1216, 512, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2(transpose_mask=True)
        ]
        return albu.Compose(aug)

    
    def to_string(logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def segmentImage(self):
        bce_loss = nn.BCEWithLogitsLoss()
        metric = Dice()
        optimizer = torch.optim.Adam
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.images.sort()
        self.masks.sort()
        test_dataset = AugmentedImageDataset(self.images, self.masks, self.get_validation_augmentation())
        seg_model = UNet(1)
        seg_model.eval()
        seg_model.load_state_dict(torch.load('Unet.h5', map_location=torch.device('cpu')))
        for i in range(len(test_dataset)):
            with torch.no_grad():
                img, label = test_dataset[i]
                imgName = self.images[i].split('\\')[-1]
                self.imgNames.append(imgName)
                pred = seg_model(img.unsqueeze(0))
                pred = torch.sigmoid(pred).detach().numpy()
            #     print(pred)
            #     pred = np.where(pred > 0.495, 1, 0)
                pred = np.where(pred > 0.51, 1.0, 0.0)
                pred = torch.from_numpy(pred)
                pred = pred.squeeze(0)

                val_logs = {}
                loss_meter = AverageValueMeter()
                metric_meter = AverageValueMeter()
                loss = bce_loss(pred, label)
                metric_val = metric(pred, label).cpu().detach().numpy()
                metric_meter.add(metric_val)
                val_logs.update({metric.__name__: metric_meter.mean})
                self.segmentDices.append(val_logs[metric.__name__])
                transform = T.ToPILImage()
                segmentedImg = transform(pred.squeeze(0))
                segmentedImg.save('result/mask/' + imgName)
        self.isSegment = True
        self.showSegmentInfo(0)