from PyQt5 import QtWidgets, uic
import sys

import cv2
from cv2 import dnn_superres

from PyQt5.uic.properties import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog

import requests # to get image from the web
import shutil # to save it locally
import os 

class MainWindow(QtWidgets.QMainWindow):      
    def __init__(self):   
        super(MainWindow, self).__init__()
        uic.loadUi('enhance.ui', self)
        self.actionadd_image.triggered.connect(self.openFileNameDialog) 
        self.checkBox.stateChanged.connect(self.show_original)
        self.enhance.clicked.connect(self.enhancez)
        self.setWindowTitle("Enhancer!")
        self.show() 
        
    def openFileNameDialog(self):
        path = QFileDialog.getOpenFileName(self, 'Open a file', '', 'Image(*.jpg *.png)')
        if path != ('', ''):
            self.path = path[0]
            self.name = os.path.basename(self.path)
        pixmap = QPixmap(self.path)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)     
        print(self.path)    
        print(self.name)
        
    def show_original(self): 
        if self.checkBox.isChecked():
            pixmap = QPixmap(self.path)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)
        else:
            pixmap = QPixmap(self.output)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)          
        
    def enhancez(self):
        self.label.clear()
        self.checkBox.setChecked(False)
        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()
        
        # Set the desired model and scale to get correct pre- and post-processing
        if self.ESPCN_x2.isChecked():
            path = "./models/ESPCN_x2.pb"
            sr.setModel("espcn", 2)
            self.output = f"./ESPCN_x2_{self.name}"           
        elif self.ESPCN_x3.isChecked():
            path = "./models/ESPCN_x3.pb"
            sr.setModel("espcn", 3)
            self.output = f"./ESPCN_x3_{self.name}"
        elif self.ESPCN_x4.isChecked():
            path = "./models/ESPCN_x4.pb"
            sr.setModel("espcn", 4)  
            self.output = f"./ESPCN_x4_{self.name}"
            
            
        elif self.FSRCNN_x2.isChecked():
            path = "./models/FSRCNN_x2.pb"
            sr.setModel("fsrcnn", 2)  
            self.output = f"./FSRCNN_x2_{self.name}"            
        elif self.FSRCNN_x3.isChecked():
            path = "./models/FSRCNN_x3.pb"
            sr.setModel("fsrcnn", 3)  
            self.output = f"./FSRCNN_x3_{self.name}"            
        elif self.FSRCNN_x4.isChecked():
            path = "./models/FSRCNN_x4.pb"
            sr.setModel("fsrcnn", 4)  
            self.output = f"./FSRCNN_x4_{self.name}"

        elif self.LapSRN_x2.isChecked():
            path = "./models/LapSRN_x2.pb"
            sr.setModel("lapsrn", 2)  
            self.output = f"./LapSRN_x2_{self.name}"            
        elif self.LapSRN_x4.isChecked():
            path = "./models/LapSRN_x4.pb"
            sr.setModel("lapsrn", 4)  
            self.output = f"./LapSRN_x4_{self.name}"            
        elif self.LapSRN_x8.isChecked():
            path = "./models/LapSRN_x8.pb"
            sr.setModel("lapsrn", 8)  
            self.output = f"./LapSRN_x8_{self.name}"
            
        # Read image
        image = cv2.imread(self.path)
        # Read the desired model    
        sr.readModel(path)
        # Upscale the image
        result = sr.upsample(image)
        print("done")
        # Save the image
        cv2.imwrite(self.output, result)
        pixmap = QPixmap(self.output)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  
    
app = 0            
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()                    