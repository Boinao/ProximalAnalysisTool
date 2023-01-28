from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
import cv2 as cv
import specdal
from scipy.stats import kruskal
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.ProsailUi import Ui_Form
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
import prosail

import matplotlib.pyplot as plt

# from . import GdalTools_utils as Utils
POSTFIX = '_Prosail_simulation'
from os import path
import seaborn as sns
import matplotlib.patches as mpatches

# pluginPath = os.path.split(os.path.dirname(__file__))[0]
# filepath = os.path.join(pluginPath, 'proximalAnalysisTool/data/prosail', 'soil_reflectance.csv')
# df = pd.read_csv(filepath, header=0, index_col=0)
# dry_soil = df[['dry_soil']].values[:,0]
# wet_soil = df[['wet_soil']].values[:,0]

from modules import Utils

class Prosail_Simulation(Ui_Form):

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
        super(Prosail_Simulation, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.lineEdit_soil_b.setEnabled(True)
        self.lineEdit_soil_w.setEnabled(True)
        self.lineEdit_min.setEnabled(False)
        self.lineEdit_max.setEnabled(False)
        self.lineEdit_step.setEnabled(False)
        self.pushButton_3.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.pushButton.clicked.connect(lambda: self.Spectra_Plot_clear())
        # self.pushButton_4.clicked.connect(lambda: self.Spectra_Plot_Stack())
        self.checkBox.toggled.connect(self.clearAllCheckboxes)
        self.CheckBox_p1.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p1) )
        self.CheckBox_p2.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p2))
        self.CheckBox_p3.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p3))
        self.CheckBox_p4.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p4))
        self.CheckBox_p5.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p5))
        self.CheckBox_p6.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p6))
        self.CheckBox_p7.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p7))
        self.CheckBox_p8.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p8))
        self.CheckBox_p9.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p9))
        self.CheckBox_p10.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p10))
        self.CheckBox_p11.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p11))
        self.CheckBox_p12.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p12))
        self.CheckBox_p13.toggled.connect(lambda: self.Prosail_variable(self.CheckBox_p13))


        self.onlyInt = QtGui.QIntValidator()
        self.onlyDou = QtGui.QDoubleValidator()

        self.lineEdit_p1.setValidator(self.onlyDou)
        self.lineEdit_p2.setValidator(self.onlyDou)
        self.lineEdit_p3.setValidator(self.onlyDou)
        self.lineEdit_p4.setValidator(self.onlyDou)
        self.lineEdit_p5.setValidator(self.onlyDou)
        self.lineEdit_p6.setValidator(self.onlyDou)
        self.lineEdit_p7.setValidator(self.onlyDou)

        self.lineEdit_p8.setValidator(self.onlyDou)
        self.lineEdit_p9.setValidator(self.onlyDou)
        self.lineEdit_p10.setValidator(self.onlyDou)
        self.lineEdit_p11.setValidator(self.onlyDou)
        self.lineEdit_p12.setValidator(self.onlyDou)
        self.lineEdit_p13.setValidator(self.onlyDou)


    def Spectra_Plot_clear(self):
        self.mplWidgetSpectral_6.clear()

    def clearAllCheckboxes(self):
        self.lineEdit_min.setEnabled(not self.checkBox.isChecked())
        self.lineEdit_max.setEnabled(not self.checkBox.isChecked())
        self.lineEdit_step.setEnabled(not self.checkBox.isChecked())

    def Prosail_variable(self, checkbox):
        self.lineEdit_p1.setEnabled(not checkbox.isChecked())
        self.lineEdit_min.setEnabled(checkbox.isChecked())
        self.lineEdit_max.setEnabled(checkbox.isChecked())
        self.lineEdit_step.setEnabled(checkbox.isChecked())


    def SpectrabrowseButton_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_3.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            self.lineEdit_3.setText("")
            return

        self.filepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit_3.setText(fname)
            self.outputFilename = os.path.dirname(fname) + "/Output" + POSTFIX
            self.lineEdit_4.setText(self.outputFilename)
            Utils.setLastUsedDir(os.path.dirname(fname))



    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir)
        if not self.outputFilename:
            return
        self.lineEdit_4.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))
        return self.outputFilename

    def call_prosail(self,n, cab, car,  cbrown, cw, cm, lai, lidfa, hspot,
                tts, tto, psi, ant, alpha, prospect_version,
                typelidf, lidfb, factor,
                rsoil0, rsoil, psoil,
                soil_spectrum1, soil_spectrum2):

        rho_canopy = prosail.run_prosail(n, cab, car,
                                                 cbrown,
                                                 cw,
                                                 cm, lai,
                                                 lidfa,
                                                 hspot, tts,
                                                 tto,
                                                 psi,
                                                 ant=ant, alpha=alpha, prospect_version=prospect_version,
                                                 typelidf=typelidf, lidfb=lidfb,
                                                 factor=factor, rsoil0=rsoil0, rsoil=rsoil,
                                                 psoil=psoil,
                                                 soil_spectrum1=soil_spectrum1, soil_spectrum2=soil_spectrum2)
        return rho_canopy

    def getMinMaxStep(self):
        min = float(self.lineEdit_min.text())
        step = float(self.lineEdit_step.text())
        max = float(self.lineEdit_max.text())

        return min, max, step


    def run(self):
        try:
            if (self.lineEdit_3.text() == ""):
                self.lineEdit_3.setFocus()
                messageDisplay = "Please Enter spectra File!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.lineEdit_4.text() == ""):
                self.lineEdit_4.setFocus()
                messageDisplay = "Cannot leave Output empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (not os.path.exists(self.lineEdit_3.text())):
                self.lineEdit_3.setFocus()
                messageDisplay = "Path does not exist : " + self.lineEdit_2.text()
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (not os.path.isdir(os.path.dirname(self.lineEdit_4.text()))):
                self.lineEdit_4.setFocus()
                messageDisplay = "Kindly enter a valid output path."
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return


            if  (self.CheckBox_p1.isChecked() or self.CheckBox_p2.isChecked() or self.CheckBox_p3.isChecked() or self.CheckBox_p4.isChecked() or
                self.CheckBox_p5.isChecked() or self.CheckBox_p6.isChecked() or self.CheckBox_p7.isChecked() or self.CheckBox_p8.isChecked() or
                self.CheckBox_p9.isChecked() or self.CheckBox_p10.isChecked() or self.CheckBox_p11.isChecked() or self.CheckBox_p12.isChecked() or
                self.CheckBox_p13.isChecked()):

                if (self.lineEdit_min.text() == ""):
                    self.lineEdit_min.setFocus()
                    messageDisplay = "Cannot leave field empty!"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                if (self.lineEdit_step.text() == ""):
                    self.lineEdit_step.setFocus()
                    messageDisplay = "Cannot leave field empty!"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                if (self.lineEdit_max.text() == ""):
                    self.lineEdit_max.setFocus()
                    messageDisplay = "Cannot leave field empty !"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return


            if (self.lineEdit_soil_b.text() == ""):
                self.lineEdit_soil_b.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.lineEdit_soil_w.text() == ""):
                self.lineEdit_soil_w.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            self.outputFilename=self.lineEdit_4.text()

            df = pd.read_csv(self.filepath, header=0, index_col=0)
            dry_soil = df[['dry_soil']].values[:,0]
            wet_soil = df[['wet_soil']].values[:,0]

            self.parameter_p1 = float(self.lineEdit_p1.text())
            self.parameter_p2 = float(self.lineEdit_p2.text())
            self.parameter_p3 = float(self.lineEdit_p3.text())
            self.parameter_p4 = float(self.lineEdit_p4.text())
            self.parameter_p5 = float(self.lineEdit_p5.text())
            self.parameter_p6 = float(self.lineEdit_p6.text())
            self.parameter_p7 = float(self.lineEdit_p7.text())
            self.parameter_p8 = float(self.lineEdit_p8.text())
            self.parameter_p9 = float(self.lineEdit_p9.text())
            self.parameter_p10 = float(self.lineEdit_p10.text())
            self.parameter_p11 = float(self.lineEdit_p11.text())
            self.parameter_p12 = float(self.lineEdit_p12.text())
            self.parameter_p13 = float(self.lineEdit_p13.text())
            self.dsoil_bright = float(self.lineEdit_soil_b.text())
            self.wsoil_bright = float(self.lineEdit_soil_w.text())
            # rho_soil = self.dsoil_bright * (self.wsoil_bright * dry_soil + (1 - self.wsoil_bright) * wet_soil)
            # print(rho_soil)
            # ------------- PRospect model -------------
            if self.radioButton.isChecked():
                prospect_model='5'
            elif self.radioButton_2.isChecked():
                prospect_model='D'
            # elif self.radioButton_3.isChecked():
            #     prospect_model='PRO'
            # --------------Reflectance type ----------------
            if self.radioButton_4.isChecked():
                Ref_type='SDR'
            elif self.radioButton_5.isChecked():
                Ref_type="BHR"
            elif self.radioButton_6.isChecked():
                Ref_type='DHR'
            elif self.radioButton_7.isChecked():
                Ref_type="HDR"
            wave = np.arange(400, 2501)

            params = [{'prosail_parameter': 'structure parameter', 'Values': self.parameter_p1}, #0
                      {'prosail_parameter': 'Chlorophyll', 'Values': self.parameter_p2}, #1
                      {'prosail_parameter': 'Carotenoid', 'Values': self.parameter_p5},  #2
                      {'prosail_parameter': 'Brown pigment', 'Values': self.parameter_p6}, #3
                      {'prosail_parameter': 'water content', 'Values': self.parameter_p3}, #4
                      {'prosail_parameter': 'Dry matter', 'Values': self.parameter_p4}, #5
                      {'prosail_parameter': 'lai', 'Values': self.parameter_p8}, #6
                      {'prosail_parameter': 'lidfa', 'Values': self.parameter_p9}, #7
                      {'prosail_parameter': 'hspot', 'Values': self.parameter_p10}, #8
                      {'prosail_parameter': 'tts', 'Values': self.parameter_p12}, #9
                      {'prosail_parameter': 'tto', 'Values': self.parameter_p11}, #10
                      {'prosail_parameter': 'psi', 'Values': self.parameter_p13}, #11
                      {'prosail_parameter': 'ant', 'Values': self.parameter_p7}, #12
                      {'prosail_parameter': 'Maxium_incident_Angle_relative_to_noraml_leaaf_plane', 'Values': 40.0}, #13
                      {'prosail_parameter': 'prospect_version', 'Values': prospect_model}, #14
                      {'prosail_parameter': "typelidf", 'Values': 2}, #15
                      {'prosail_parameter': "lidfb", 'Values': 0.0}, #16
                      {'prosail_parameter': 'factor', 'Values': Ref_type}, #17
                      {'prosail_parameter': 'rsoil', 'Values': self.dsoil_bright}, #18
                      {'prosail_parameter': 'psoil', 'Values': self.wsoil_bright}] #19

            default_param_list=[self.parameter_p1,self.parameter_p2, self.parameter_p5,
                                self.parameter_p6,self.parameter_p3, self.parameter_p4, self.parameter_p8,
                                self.parameter_p9, self.parameter_p10, self.parameter_p12,
                                self.parameter_p11,self.parameter_p13, self.parameter_p7, 40.0, prospect_model,
                                2,0.0, Ref_type, None, self.dsoil_bright, self.wsoil_bright, dry_soil, wet_soil]

            index=-1
            if self.CheckBox_p1.isChecked():
                index = 0
                self.min, self.max, self.step=self.getMinMaxStep()
                if (self.min < 1.0) or (self.max > 3.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 1 to 3"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[0] = {'prosail_parameter': 'structure parameter',
                             'Values': np.arange(self.min, self.max, self.step)}

            elif self.CheckBox_p2.isChecked():
                index = 1
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 1.0) or (self.max > 100.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 1 to 100"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                params[index] = {'prosail_parameter': 'Chlorophyll', 'Values': np.arange(self.min, self.max, self.step)}

            elif self.CheckBox_p5.isChecked():
                index = 2
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 30) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0 to 30"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                params[2] = {'prosail_parameter': 'Carotenoid', 'Values': np.arange(self.min, self.max, self.step)}
            elif self.CheckBox_p6.isChecked():
                index = 3
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 1.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0.0 to 1.0"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[3] = {'prosail_parameter': 'Brown pigment', 'Values': np.arange(self.min, self.max, self.step)}


            elif self.CheckBox_p3.isChecked():
                index = 4
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0002) or (self.max > 0.2) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0.0002 to 0.2"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[4] = {'prosail_parameter': 'water content', 'Values': np.arange(self.min, self.max, self.step)}

            elif self.CheckBox_p4.isChecked():
                index = 5
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0019) or (self.max > 0.0165) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0.0019 to 0.0165"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                params[5] ={'prosail_parameter': 'Dry matter', 'Values': np.arange(self.min, self.max, self.step)}

            elif self.CheckBox_p8.isChecked():
                index = 6
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.01) or (self.max > 10.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0.01 to 10"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[6] = {'prosail_parameter': 'lai', 'Values': np.arange(self.min, self.max, self.step)}
            elif self.CheckBox_p9.isChecked():
                index = 7
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 90.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0.0 to 90"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[7] = {'prosail_parameter': 'lidfa', 'Values': np.arange(self.min, self.max, self.step)}

            elif self.CheckBox_p10.isChecked():
                index = 8
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 1.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number between 0.0 to 1.0"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[8] = {'prosail_parameter': 'hspot', 'Values': np.arange(self.min, self.max, self.step)}

            elif self.CheckBox_p12.isChecked():
                index = 9
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 89.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0 to 89"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[9] = {'prosail_parameter': 'tts', 'Values': np.arange(self.min, self.max, self.step)}

            elif self.CheckBox_p11.isChecked():
                index = 10
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 89.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0 to 89"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[10] = {'prosail_parameter': 'tto', 'Values': np.arange(self.min, self.max, self.step)}



            elif self.CheckBox_p13.isChecked():
                index = 11
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 180.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0 to 180"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
                params[11] = {'prosail_parameter': 'psi', 'Values': np.arange(self.min, self.max, self.step)}


            elif self.CheckBox_p7.isChecked():
                index = 12
                self.min, self.max, self.step = self.getMinMaxStep()
                if (self.min < 0.0) or (self.max > 10.0) or self.min > self.max or self.min == self.max:
                    messageDisplay = "please enther the number 0.0 to 10.0"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                params[12] = {'prosail_parameter': 'ant', 'Values': np.arange(self.min, self.max, self.step)}

            col = ['Spectrum']
            if index >1:
                n_sample = np.arange(self.min, self.max, self.step).shape
                spectra = np.zeros((len(wave), n_sample[0]))
                count = 0

                for i in np.arange(self.min, self.max, self.step):
                    default_param_list[index] = i
                    # param_list.append(default_param_list)
                    rho_canopy = self.call_prosail(*default_param_list)
                    self.mplWidgetSpectral_6.ax.plot(wave, rho_canopy)

                    spectra[:, count] = rho_canopy
                    count = count + 1
                self.mplWidgetSpectral_6.canvas.draw()
                col = ["Sample " + str(i) for i in np.arange(0, n_sample[0])]
            else:

                if 3.0 >= self.parameter_p1 >= 1.0 and 100 >= self.parameter_p2 >= 1.0 and 0.2 >= self.parameter_p3 >= 0.0002 \
                        and 0.0165 >= self.parameter_p4 >= 0.0019 and 30 >= self.parameter_p5 >= 0 and 1.0 >= self.parameter_p6 >= 0 \
                        and 10 >= self.parameter_p7 >= 0 and 10 >= self.parameter_p8 >= 0.01 \
                        and 90 >= self.parameter_p9 >= 0.0 and 1.0 >= self.parameter_p10 >= 0.0 \
                        and 89 >= self.parameter_p11 >= 0.0 and 89 >= self.parameter_p12 >= 0.0 and 180 >= self.parameter_p13 >= 0.0:

                    spectra = self.call_prosail(*default_param_list)
                    self.mplWidgetSpectral_6.ax.plot(wave, spectra)
                else:
                    messageDisplay = "please enter the parameter within the limit"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return


            self.mplWidgetSpectral_6.ax.set_title("Prosail Simulation")
            self.mplWidgetSpectral_6.ax.set_xlabel("Wavelength")
            self.mplWidgetSpectral_6.ax.set_ylabel("Reflectance")
            self.mplWidgetSpectral_6.canvas.draw()

            fields = ['prosail_parameter', 'Values']
            # Save csv
            row = wave
            df1 = pd.DataFrame(spectra, index=row, columns=col)
            df1.to_csv(self.outputFilename + '_Spectra' + '.csv', header=True, index=True)
            import csv
            with open(self.outputFilename + '_Params' + '.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                writer.writerows(params)



        except Exception as e:
            import traceback
            print(e, traceback.format_exc())
            QApplication.restoreOverrideCursor()
