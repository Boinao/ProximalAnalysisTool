# -*- coding: utf-8 -*-

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
import cv2 as cv
import specdal
from scipy.stats import kruskal
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from Ui.Spectra_simulation_SACUi import Ui_Form
import os
import math
import scipy.stats as stats
from math import exp, sqrt, log
from PIL import Image
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import pysptools.spectro as spectro
from modules.PandasModel import PandasModel
# Import the Py6S module
from Py6S import SixS

import matplotlib.pyplot as plt

# from . import GdalTools_utils as Utils
POSTFIX = '_Spectra_simulation'
from os import path
import seaborn as sns
import matplotlib.patches as mpatches

from modules import Utils

class SimulationMixing(Ui_Form):

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
        super(SimulationMixing, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())

        self.pushButton.clicked.connect(lambda: self.clear_plot())

        self.onlyInt = QIntValidator()
        self.onlyDou = QDoubleValidator()

        self.x1CoordEdit_11.setValidator(QDoubleValidator(0.0,1.0,2))
        self.x1CoordEdit_12.setValidator(QDoubleValidator(0.0,1.0,2))
        self.x1CoordEdit_13.setValidator(QDoubleValidator(0.0,1.0,2))
        self.x1CoordEdit_14.setValidator(QDoubleValidator(0.0,1.0,2))

    def clear_plot(self):
        self.mplWidgetSpectral_5.clear()
        # print("Clear")

    def SpectrabrowseButton_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()
        self.lineEdit_2.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            self.lineEdit_3.setText("")
            return

        self.spectra_filepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit_2.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))

            self.outputFilename = os.path.dirname(fname)+  "/Output" + POSTFIX + ".csv"
            self.lineEdit_3.setText(self.outputFilename)
        # else:
        #     self.lineEdit_2.setText("")
    def saveas(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')
        if not self.outputFilename:
            return

        self.lineEdit_3.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename

    def Linear_endmember(self, E, a1, wavelength):
        M = np.size(E, 1);
        a2 = 1 - a1
        num1 = 0;
        F = []
        indxk = []
        indxj = []
        for k in range(M - 1):
            for j in range(k + 1, M):
                indxk.append(k)
                indxj.append(j)
                F.append(a1 * E[:, k] + a2 * E[:, j])
                num1 = num1 + 1;
        F = np.asarray(F).T
        self.mplWidgetSpectral_5.clear()
        self.mplWidgetSpectral_5.ax.plot(wavelength,F)
        self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
        self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
        self.mplWidgetSpectral_5.canvas.draw()
        return F, indxk, indxj


    def getBilinearComponent(self,E,const):
        M = np.size(E, 1);
        num1 = 0;
        F = []
        indxk = []
        indxj = []
        for k in range(M - 1):
            for j in range(k + 1, M):
                indxk.append(k)
                indxj.append(j)
                F.append(const * (E[:, k] * E[:, j]))
                num1 = num1 + 1;
        return F, indxk, indxj


    def Bilinear_endmember(self, E, const,wavelength):

        F, indxk, indxj=self.getBilinearComponent(E,const)
        F = np.asarray(F).T
        self.mplWidgetSpectral_5.clear()
        self.mplWidgetSpectral_5.ax.plot(wavelength,F)
        self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
        self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
        self.mplWidgetSpectral_5.canvas.draw()
        return F, indxk, indxj

    def Fans(self, E, const, a1):
        M = np.size(E, 1);
        a2 = 1 - a1
        num1 = 0;
        L = []
        indxk = []
        indxj = []
        for k in range(M - 1):
            for j in range(k + 1, M):
                indxk.append(k)
                indxj.append(j)
                L.append(a1 * E[:, k] + a2 * E[:, j])
                num1 = num1 + 1;
        L = np.asarray(L).T
        M = np.size(E, 1);
        num1 = 0;

        F, indxk, indxj = self.getBilinearComponent(E, const)
        F = np.asarray(F).T
        fans = L + F
        self.mplWidgetSpectral_5.clear()
        self.mplWidgetSpectral_5.ax.plot(fans)
        self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
        self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
        self.mplWidgetSpectral_5.canvas.draw()
        return fans, indxk, indxj

    def run(self):

        if (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            messageDisplay = "Cannot leave field empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_2.text())):
            self.lineEdit_2.setFocus()
            messageDisplay = "Path does not exist : " + self.lineEdit_2.text()
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_3.text() == ""):
            self.lineEdit_3.setFocus()
            messageDisplay = "Cannot leave field empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.isdir(os.path.dirname(self.lineEdit_3.text()))):
            self.lineEdit_3.setFocus()
            messageDisplay = "Kindly enter a valid output path."
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if self.Linear_mixing_button.isChecked():
            if (self.x1CoordEdit_11.text() == ""):
                self.x1CoordEdit_11.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (float(self.x1CoordEdit_11.text())>1.0 or float(self.x1CoordEdit_11.text())<0.0):
                self.x1CoordEdit_11.setFocus()
                messageDisplay = "Fractional value entered is Out of range !"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        if self.radioButton.isChecked():
            if (self.x1CoordEdit_12.text() == ""):
                self.x1CoordEdit_12.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (float(self.x1CoordEdit_12.text())>1.0 or float(self.x1CoordEdit_12.text())<0.0):
                self.x1CoordEdit_12.setFocus()
                messageDisplay = "Constant value entered is Out of range !"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        if self.radioButton_2.isChecked():
            if (self.x1CoordEdit_13.text() == ""):
                self.x1CoordEdit_13.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_14.text() == ""):
                self.x1CoordEdit_14.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (float(self.x1CoordEdit_13.text()) > 1.0 or float(self.x1CoordEdit_13.text()) < 0.0):
                self.x1CoordEdit_13.setFocus()
                messageDisplay = "Fractional value entered is Out of range !"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (float(self.x1CoordEdit_14.text()) > 1.0 or float(self.x1CoordEdit_14.text()) < 0.0):
                self.x1CoordEdit_14.setFocus()
                messageDisplay = "Constant value entered is Out of range !"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return



        try:

            self.outputFilename=self.lineEdit_3.text()

            data_spectra = pd.read_csv(self.spectra_filepath, header=0, index_col=0)
            self.spectra = data_spectra.to_numpy()
            wavelength = data_spectra.index.values


            Linear_fraction = float(self.x1CoordEdit_11.text())
            Bilinear_constant = float(self.x1CoordEdit_12.text())
            Fans_fraction = float(self.x1CoordEdit_13.text())
            Fans_constant = float(self.x1CoordEdit_14.text())
            if self.Linear_mixing_button.isChecked():
                # print('hello')
                L, indi, indj = self.Linear_endmember(self.spectra, Linear_fraction, wavelength)
                # print(L.shape,len(indi))
                stacked=np.hstack((np.asarray(indi)[:,np.newaxis],np.asarray(indj)[:,np.newaxis],L.T))
                # print(stacked.shape)
                ind = ['Sample Index (A)', 'Sample Index (B)']+[str(i) for i in wavelength]#range(L.shape[0])]
                res = pd.DataFrame(stacked, columns=ind)
                res.to_csv(self.outputFilename, index=False)
            if self.radioButton.isChecked():
                F, indi, indj = self.Bilinear_endmember(self.spectra, Bilinear_constant, wavelength);
                stacked = np.hstack((np.asarray(indi)[:, np.newaxis], np.asarray(indj)[:, np.newaxis], F.T))
                # print(stacked.shape)
                ind = ['Sample Index (A)', 'Sample Index (B)'] + [str(i) for i in  wavelength]#range(F.shape[0])]
                res = pd.DataFrame(stacked,columns=ind)
                res.to_csv(self.outputFilename, index=False)
            if self.radioButton_2.isChecked():
                L, indi, indj = self.Linear_endmember(self.spectra, Fans_fraction,wavelength)
                F, indi, indj = self.Bilinear_endmember(self.spectra, Fans_constant, wavelength);
                Fans = L + F
                stacked = np.hstack((np.asarray(indi)[:, np.newaxis], np.asarray(indj)[:, np.newaxis], Fans.T))
                # print(stacked.shape)
                ind = ['Sample Index (A)', 'Sample Index (B)'] + [str(i) for i in wavelength]#range(L.shape[0])]
                res = pd.DataFrame(stacked, columns=ind)
                res.to_csv(self.outputFilename, index=False)

        except Exception as e:
            QApplication.restoreOverrideCursor()
            print(e)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    # QSizePolicy sretain=Form.sizePolicy()
    # sretain.setRetainSizeWhenHidden(True)
    # sretain.setSizePolicy()
    ui = SpectraSimulation()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
