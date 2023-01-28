# -*- coding: utf-8 -*-
"""
Created on Mon January 4 11:05:06 2021

@author: Nidhin

"""
from PyQt5 import QtCore
from PyQt5 import QtWidgets,QtGui
import cv2 as cv
import specdal
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.FgccUi import Ui_Form
import os
from PIL import Image
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from modules.PandasModel import PandasModel
import colorama
from colorama import Fore
import matplotlib.pyplot as plt
import collections
import math
POSTFIX = '_Fgcc'
from os import path

from modules import Utils
class Fgcc(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.outputFilename = ""
        self.data = pd.DataFrame()


    def get_widget(self):
        return self.groupBox

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def setupUi(self, Form):
        super(Fgcc, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()
        self.mplWidget.setVisible(False)
        self.mplWidget_2.setVisible(False)

    def connectWidgets(self):
        self.lineEdit.setEnabled(True)
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        # To allow only int
        self.onlyInt = QtGui.QIntValidator()
        self.onlyDou = QtGui.QDoubleValidator()
        self.lineEdit.setValidator(self.onlyInt)
        self.lineEdit_7.setValidator(self.onlyInt)
        self.lineEdit_8.setValidator(self.onlyInt)
        self.lineEdit_2.setValidator(self.onlyDou)
        self.lineEdit_5.setValidator(self.onlyDou)
        self.lineEdit_6.setValidator(self.onlyDou)

        # self.radioButton.toggled.connect(lambda: self.stateChanged1())
        # self.radioButton_2.toggled.connect(lambda: self.stateChanged2())

    # def stateChanged1(self):
    #     self.lineEdit.setEnabled(True)
    #     self.lineEdit_6.setEnabled(True)
    #     self.lineEdit_5.setEnabled(True)
    #     self.lineEdit_2.setEnabled(True)

    # def stateChanged2(self):
    #     self.lineEdit.setEnabled(True)
    #     self.lineEdit_6.setEnabled(True)
    #     self.lineEdit_5.setEnabled(True)
    #     self.lineEdit_2.setEnabled(True)

    def browseButton_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_3.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.jpg *.png)", directory=lastDataDir)

        if not fname:
            self.lineEdit_3.setText("")
            self.lineEdit_4.setText("")
            return

        self.filepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit_3.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))

            self.outputFilename = os.path.dirname(fname)+ "/Output" + POSTFIX + ".png"
            self.metadatafile = os.path.dirname(fname) + "/MetadataOutput" + POSTFIX + ".csv"
            self.lineEdit_4.setText(self.outputFilename)


    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.png')
        if not self.outputFilename:
            return

        self.lineEdit_4.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename

    def run(self):
        self.mplWidget.ax.clear()

        if not (Utils.validataInputPath(self.lineEdit_3, self.Form) and \
                Utils.validataOutputPath(self.lineEdit_4, self.Form) and \
                Utils.validateExtension(self.lineEdit_4, 'png', self.Form)
                ):
            return



        if (self.lineEdit.text() == ""):
            self.lineEdit.setFocus()
            messageDisplay = "Please enter Radius!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_7.text() == ""):
            self.lineEdit_7.setFocus()
            messageDisplay = "Please enter distance!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_8.text() == ""):
            self.lineEdit_8.setFocus()
            messageDisplay = "Please enter pixel size!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            messageDisplay = "Cannot leave Red Green value empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_5.text() == ""):
            self.lineEdit_5.setFocus()
            messageDisplay = "Cannot leave Blue Green value empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_6.text() == ""):
            self.lineEdit_6.setFocus()
            messageDisplay = "Cannot leave RGB value empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        self.outputFilename = self.lineEdit_4.text()
        self.req_angle = float(self.lineEdit.text())
        self.req_distance = float(self.lineEdit_7.text())
        self.req_pixels = float(self.lineEdit_8.text())
        self.req_RG = float(self.lineEdit_2.text())
        self.req_BG = float(self.lineEdit_5.text())
        self.req_RGB = float(self.lineEdit_6.text())



        self.textBrowser_2.clear()
        imgtoproc2 = Image.open(self.filepath).convert("RGB")
        # self.mplWidget.ax.imshow(imgtoproc2)
        # self.mplWidget.setXAxisCaption("")
        # self.mplWidget.setYAxisCaption("")
        # self.mplWidget.canvas.draw()
        im = np.array(imgtoproc2)
        r, c, d = im.shape
        # print(r, c)
        imaged = cv.cvtColor(im, cv.COLOR_RGB2HSV)
        # img = rgb2gray(im)
        circ_img = np.zeros((r, c), np.uint8)
        self.req_radius = (float(math.tan((self.req_angle / 2)*np.pi/180) * self.req_distance))
        self.pix_radius = int(self.req_radius*self.req_pixels)
        cv.circle(circ_img, (c // 2, r // 2), self.pix_radius, 1, thickness=-1)
        masked_data = cv.bitwise_and(im, im, mask=circ_img)
        R = masked_data[:, :, 0]
        G = masked_data[:, :, 1]
        B = masked_data[:, :, 2]
        RG = np.divide(R, G, where=G != 0)
        BG = np.divide(B, G, where=G != 0)
        RGB = 2 * G - R - B
        data = RGB
        # res = 255 * (data - np.min(data) / (np.max(data) - np.min(data)))
        green = (RG < self.req_RG) & (BG < self.req_BG) & (RGB > self.req_RGB) * 1
        grayr = masked_data[:, :, 0].reshape(r * c, 1)
        mask = (grayr < 0.1) * 1
        Tot = r * c
        maskp = Tot - np.count_nonzero(mask)
        nongreenpx = maskp - np.count_nonzero(green)
        greenpx = maskp - nongreenpx
        per_green = 100 * greenpx / (nongreenpx + greenpx)

        # print(str(self.filepath.split('/')[-1].split('.')[0]) + ":", per_green)
        self.textBrowser_2.append(str(per_green))
        # METADATA GENERATION
        self.metadata = collections.namedtuple("metadata","radius RedGreen BlueGreen RGB")
        self.meta = self.metadata(self.pix_radius,self.req_RG,self.req_BG,self.req_RGB)
        save = self.meta._asdict()
        # print(save,'meta')
        metadata = pd.DataFrame(save, index=[0])
        metadata.to_csv(self.metadatafile, header=True, index=True)
        # self.mplWidget_2.ax.imshow(masked_data)
        # self.mplWidget_2.ax.figure.savefig(self.outputFilename)
        # self.mplWidget_2.setXAxisCaption("")
        # self.mplWidget_2.setYAxisCaption("")
        # self.mplWidget.ax.imshow(green, cmap="gray")
        # self.mplWidget.ax.figure.savefig(self.outputFilename)
        # self.mplWidget.setXAxisCaption("")
        # self.mplWidget.setYAxisCaption("")
        # self.mplWidget_2.canvas.draw()
        # self.mplWidget.canvas.draw()

        plt.figure()
        ax1 = plt.subplot(131)
        ax1.imshow(im)
        ax1.set_title('Input Image')
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2=plt.subplot(132,sharex=ax1, sharey=ax1)
        ax2.imshow(masked_data)
        ax2.set_title('Masked Image')
        ax2.set_xticks([])
        ax2.set_yticks([])
        # plt.figure()
        ax3=plt.subplot(133,sharex=ax1, sharey=ax1)
        ax3.imshow(green, cmap='gray')
        ax3.set_title('Green Coverage')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.savefig(self.outputFilename)
        #
        # plt.plot(coeffs[0].T)
        # plt.title('Approximation Level ' + str(level))
        # l = level
        # for i in range(1, level + 1):
        #     plt.subplot(2, int(np.ceil(level / 2)) + 1, i + 1)
        #     plt.plot(coeffs[i].T)
        #     plt.title("Details Level " + str(l))
        #     l = l - 1

        plt.show()

        # elif self.radioButton_2.isChecked():
        #     self.textBrowser_2.clear()
        #     imgtoproc2 = Image.open(self.filepath).convert("RGB")
        #     im = np.array(imgtoproc2)
        #     r, c, d = im.shape
        #     # print(r, c)
        #     imaged = cv.cvtColor(im, cv.COLOR_RGB2HSV)
        #     # img = rgb2gray(im)
        #     circ_img = np.zeros((r, c), np.uint8)
        #     self.req_radius = (float(math.tan((self.req_angle / 2)*np.pi/180) * self.req_distance))
        #     self.pix_radius = int(self.req_radius * self.req_pixels)
        #     cv.circle(circ_img, (c // 2, r // 2), self.pix_radius, 1, thickness=-1)
        #     masked_data = cv.bitwise_and(im, im, mask=circ_img)
        #     R = masked_data[:, :, 0]
        #     G = masked_data[:, :, 1]
        #     B = masked_data[:, :, 2]
        #     RG = np.divide(R, G, where=G != 0)
        #     BG = np.divide(B, G, where=G != 0)
        #     RGB = 2 * G - R - B
        #     data = RGB
        #     # res = 255 * (data - np.min(data) / (np.max(data) - np.min(data)))
        #     green = (RG < self.req_RG) & (BG < self.req_BG) & (RGB > self.req_RGB) * 1
        #     # plt.imshow(green, cmap='gray')
        #     # plt.savefig(self.outputFilename)
        #     # plt.show()
        #     # self.mplWidget.canvas.ax.set_aspect("auto")
        #     # METADATA GENERATION
        #     self.metadata = collections.namedtuple("metadata", "radius RedGreen BlueGreen RGB")
        #     self.meta = self.metadata(self.pix_radius, self.req_RG, self.req_BG, self.req_RGB)
        #     save = self.meta._asdict()
        #     print(save, 'meta')
        #     metadata = pd.DataFrame(save, index=[0])
        #     metadata.to_csv(self.metadatafile, header=True, index=True)
        #     self.mplWidget.ax.imshow(green,cmap="gray")
        #     self.mplWidget.ax.figure.savefig(self.outputFilename)
        #     self.mplWidget.setXAxisCaption("")
        #     self.mplWidget.setYAxisCaption("")
        # self.mplWidget.canvas.draw()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    # QSizePolicy sretain=Form.sizePolicy()
    # sretain.setRetainSizeWhenHidden(True)
    # sretain.setSizePolicy()
    ui = Fgcc()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
