# -*- coding: utf-8 -*-
"""
Created on Mon January 4 11:05:06 2021

@author: Nidhin
"""
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5 import QtCore,QtGui
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import cv2 as cv
import specdal
from scipy.stats import kruskal
from Ui.Py6sUi import Ui_Form
import os
import math
import scipy.stats as stats
from math import exp, sqrt, log
from PIL import Image
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import pysptools.spectro as spectro

import prosail
from modules.PandasModel import PandasModel
# Import the Py6S module
from Py6S import SixS, AtmosProfile, AeroProfile,GroundReflectance, SixSHelpers, Wavelength
# s = SixS()
import matplotlib.pyplot as plt

# from . import GdalTools_utils as Utils
POSTFIX = '_Py6s'
from os import path
import seaborn as sns
import matplotlib.patches as mpatches

import sys

from modules import Utils

pluginPath = os.path.split(os.path.dirname(__file__))[0]
sys.path.insert(0, os.path.join(pluginPath,"external"))
class Py6s(Ui_Form):

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
        super(Py6s, self).setupUi(Form)
        self.Form = Form
        self.label_73.setVisible(False)
        self.lineEdit_6.setVisible(False)
        self.pushButton_5.setVisible(False)
        # label_73 lineEdit_6 pushButton_5
        self.connectWidgets()

    def connectWidgets(self):

        self.checkBox_29.toggled.connect(self.Dis_Atmos_profile)
        self.checkBox_57.toggled.connect(self.Dis_Atmos_profile_user)
        self.checkBox_select.toggled.connect(self.input_select)
        self.pushButton_4.clicked.connect(lambda: self.target_spec_clicked())
        self.pushButton_5.clicked.connect(lambda: self.background_spec_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())

        self.onlyInt = QtGui.QIntValidator()
        self.onlyDou = QtGui.QDoubleValidator()
        self.x1CoordEdit_272.setValidator(self.onlyDou)
        self.x1CoordEdit_273.setValidator(self.onlyDou)
        self.x1CoordEdit_274.setValidator(self.onlyDou)
        self.x1CoordEdit_275.setValidator(self.onlyDou)
        self.x1CoordEdit_276.setValidator(self.onlyDou)
        self.x1CoordEdit_262.setValidator(self.onlyDou)
        self.x1CoordEdit_263.setValidator(self.onlyDou)

        self.x1CoordEdit_264.setValidator(self.onlyDou)
        self.x1CoordEdit_265.setValidator(self.onlyDou)
        self.x1CoordEdit_266.setValidator(self.onlyDou)
        self.x1CoordEdit_267.setValidator(self.onlyDou)
        self.x1CoordEdit_268.setValidator(self.onlyDou)
        self.x1CoordEdit_269.setValidator(self.onlyDou)
        self.x1CoordEdit_270.setValidator(self.onlyDou)
        self.x1CoordEdit_271.setValidator(self.onlyDou)

        self.x1CoordEdit_293.setValidator(self.onlyDou)
        self.x1CoordEdit_297.setValidator(self.onlyDou)
        self.x1CoordEdit_298.setValidator(self.onlyDou)

        self.x1CoordEdit_299.setValidator(self.onlyDou)
        self.x1CoordEdit_300.setValidator(self.onlyDou)
        self.x1CoordEdit_301.setValidator(self.onlyDou)
        self.x1CoordEdit_302.setValidator(self.onlyDou)
        self.x1CoordEdit_303.setValidator(self.onlyDou)
        self.x1CoordEdit_304.setValidator(self.onlyDou)
        self.x1CoordEdit_305.setValidator(self.onlyDou)
        self.x1CoordEdit_306.setValidator(self.onlyDou)



    def Dis_Atmos_profile(self):
        self.x1CoordEdit_262.setEnabled(self.checkBox_29.isChecked())
        self.x1CoordEdit_263.setEnabled(self.checkBox_29.isChecked())

    def Dis_Atmos_profile_user(self):
        self.x1CoordEdit_297.setEnabled(self.checkBox_57.isChecked())
        self.x1CoordEdit_298.setEnabled(self.checkBox_57.isChecked())


    def input_select(self):
        self.lineEdit_3.setEnabled(self.checkBox_select.isChecked())
        self.lineEdit_6.setEnabled(self.checkBox_select.isChecked())
        self.lineEdit_5.setEnabled(self.checkBox_select.isChecked())
        self.pushButton_4.setEnabled(self.checkBox_select.isChecked())
        self.pushButton_5.setEnabled(self.checkBox_select.isChecked())
        self.pushButton_6.setEnabled(self.checkBox_select.isChecked())

    def target_spec_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_5.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            self.lineEdit_5.setText("")
            return


        self.filepath1 = fname

        # print(self.filepath)
        if fname:
            # inputText = str(fname[0]) + " "
            # for i in range(1, len(fname)):
            #     inputText = inputText + " " + fname[i]
            self.lineEdit_5.setText(fname)

            self.outputFilename = os.path.dirname(fname) + "/Output" + POSTFIX + ".csv"
            self.lineEdit_3.setText(self.outputFilename)
            Utils.setLastUsedDir(os.path.dirname(fname))


    def background_spec_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_6.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            self.lineEdit_6.setText("")
            return


        self.filepath2 = fname

        if fname:
            # inputText = str(fname[0]) + " "
            # for i in range(1, len(fname)):
            #     inputText = inputText + " " + fname[i]
            self.lineEdit_6.setText(fname)

            self.outputFilename = os.path.dirname(fname) + "/Output" + POSTFIX + ".csv"
            self.lineEdit_3.setText(self.outputFilename)
            Utils.setLastUsedDir(os.path.dirname(fname))


    def saveas(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')
        if not self.outputFilename:
            return

        self.lineEdit_3.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename

    def run(self):

        if self.tabWidget_4.currentIndex() == 0:

            if (self.x1CoordEdit_272.text() == ""):
                self.x1CoordEdit_272.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_273.text() == ""):
                self.x1CoordEdit_273.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_274.text() == ""):
                self.x1CoordEdit_274.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_275.text() == ""):
                self.x1CoordEdit_275.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_276.text() == ""):
                self.x1CoordEdit_276.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if self.checkBox_29.isChecked():

                if (self.x1CoordEdit_262.text() == ""):
                    self.x1CoordEdit_262.setFocus()
                    messageDisplay = "Cannot leave field empty!"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                if (self.x1CoordEdit_263.text() == ""):
                    self.x1CoordEdit_263.setFocus()
                    messageDisplay = "Cannot leave field empty!"
                    QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

            if (self.x1CoordEdit_264.text() == ""):
                self.x1CoordEdit_264.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_265.text() == ""):
                self.x1CoordEdit_265.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_266.text() == ""):
                self.x1CoordEdit_266.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return


            if (self.x1CoordEdit_267.text() == ""):
                self.x1CoordEdit_267.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return


            if (self.x1CoordEdit_268.text() == ""):
                self.x1CoordEdit_268.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_269.text() == ""):
                self.x1CoordEdit_269.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_270.text() == ""):
                self.x1CoordEdit_270.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_271.text() == ""):
                self.x1CoordEdit_271.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
        else:

            if not self.checkBox_select.isChecked():
                self.checkBox_select.setFocus()
                messageDisplay = "Please activate the checkbox first!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return


            if (self.lineEdit_5.text() == ""):
                self.lineEdit_5.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return


            if (self.lineEdit_3.text() == ""):
                self.lineEdit_3.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_293.text() == ""):
                self.x1CoordEdit_293.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_297.text() == ""):
                self.x1CoordEdit_297.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_298.text() == ""):
                self.x1CoordEdit_298.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_299.text() == ""):
                self.x1CoordEdit_299.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_300.text() == ""):
                self.x1CoordEdit_300.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_301.text() == ""):
                self.x1CoordEdit_301.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_302.text() == ""):
                self.x1CoordEdit_302.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_303.text() == ""):
                self.x1CoordEdit_303.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_304.text() == ""):
                self.x1CoordEdit_304.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_305.text() == ""):
                self.x1CoordEdit_305.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.x1CoordEdit_306.text() == ""):
                self.x1CoordEdit_306.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        filepath = os.path.join(pluginPath, 'external', 'sixs.exe')
        if self.checkBox_select.isChecked():

            self.filepath1=self.lineEdit_5.text()
            self.outputFilename=self.lineEdit_3.text()
            try:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                QApplication.processEvents()
                df_spectra = pd.read_csv(self.filepath1,header=0, index_col=None)
                # df_spectra=df_spectra.dropna()
                spectra1 = df_spectra.to_numpy().reshape(2101,2)

                self.radius=float(self.x1CoordEdit_293.text())
                self.solar_zenith_p3 = float(self.x1CoordEdit_299.text())
                self.solar_azimuth_p4 = float(self.x1CoordEdit_300.text())
                self.view_zenith_p5 = float(self.x1CoordEdit_301.text())
                self.view_azimuth_p6 = float(self.x1CoordEdit_302.text())
                self.month_p7 = float(self.x1CoordEdit_303.text())
                self.day_p8 = float(self.x1CoordEdit_304.text())
                self.sensor_altitude_p9 = float(self.x1CoordEdit_305.text())
                self.target_alt_p10 = float(self.x1CoordEdit_306.text())
                if self.radioPixel_radiance_3.isChecked():
                    type_pixel = 'pixel_radiance'
                elif self.radioPixel_reflectance_3.isChecked():
                    type_pixel = 'pixel_reflectance'

                s = SixS(path=filepath)



                if not isinstance(s, SixS):
                    print('Cannot instantiate SixS')

                # ------------- Atmospheric profile------------------------
                if self.checkBox_57.isChecked():
                    self.water = float(self.x1CoordEdit_297.text())
                    self.ozone = float(self.x1CoordEdit_298.text())
                    s.atmos_profile = AtmosProfile.UserWaterAndOzone(self.water, self.ozone)

                elif self.checkBox_58.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)
                elif self.checkBox_59.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)
                elif self.checkBox_60.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
                elif self.checkBox_61.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter)
                elif self.checkBox_62.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticSummer)
                else:
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticWinter)

                # ------------- Aero profile------------------------
                if self.checkBox_64.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
                elif self.checkBox_65.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Continental)
                elif self.checkBox_66.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Maritime)
                elif self.checkBox_67.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Urban)
                elif self.checkBox_68.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Desert)
                elif self.checkBox_69.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.BiomassBurning)
                else:
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Stratospheric)

                # ------------- Run Simulation------------------------
                print('Simulation Running...')
                # s.wavelength = Wavelength(self.Wavelength_p1)
                # HomogeneousWalthall(0.48, 0.50, 2.95, 0.6)#  0.0512 0.3169 -0.3085 0.065HomogeneousWalthall(0.0512, 0.3169, -0.3085, 0.065)#HomogeneousRoujean(0.37, 0.0, 0.133)#
                s.geometry.solar_z = self.solar_zenith_p3
                s.geometry.solar_a = self.solar_azimuth_p4
                s.geometry.view_z = self.view_zenith_p5
                s.geometry.view_a = self.view_azimuth_p6
                s.geometry.day = self.month_p7
                s.geometry.month = self.day_p8
                s.altitudes.set_sensor_satellite_level = self.sensor_altitude_p9
                s.altitudes.set_target_sea_level = self.target_alt_p10
                s.ground_reflectance = GroundReflectance.HomogeneousLambertian(spectra1) #HeterogeneousLambertian(self.radius, spectra1, spectra2)
                wavelengths, results = SixSHelpers.Wavelengths.run_whole_range(s,output_name=type_pixel)
                data={'wavelengths':wavelengths,'results':results}
                df=pd.DataFrame(data)
                df.to_csv(self.outputFilename, index=False)

                # print(wavelengths, results)
                self.mplWidgetSpectral_15.ax.clear()
                self.mplWidgetSpectral_15.ax.plot(wavelengths,results,label='TOA Spectrum')
                self.mplWidgetSpectral_15.ax.plot(spectra1[:,0], spectra1[:,1],label='Input Spectrum')
                self.mplWidgetSpectral_15.ax.set_xlabel('Wavelength')
                self.mplWidgetSpectral_15.ax.set_ylabel(type_pixel)
                self.mplWidgetSpectral_15.ax.legend()
                self.mplWidgetSpectral_15.canvas.draw()
                plt.show()

                # SixSHelpers.Angles.run_and_plot_360(s, 'solar', output_name=type)
                # results, azimuths, zeniths, s.geometry.solar_a, s.geometry.solar_z = SixSHelpers.Angles.run360(s, 'view', output_name=type)
                # self.plot_polar_contour(results, azimuths, zeniths, filled=True)

                # data.to_csv('D:/test.csv')
                # self.mplWidgetSpectral_19.ax.plot(ax)
                # self.mplWidgetSpectral_19.ax.colorbar()

                # self.mplWidgetSpectral_19.canvas.draw()
                print('Simulation Completed Successfully...')
                QApplication.restoreOverrideCursor()
            except Exception as e:
                import traceback
                print(e, traceback.format_exc())
                QApplication.restoreOverrideCursor()
        else:
            try:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                QApplication.processEvents()
                self.Wavelength_p1 = float(self.x1CoordEdit_272.text())
                self.Reflectance_p2 = float(self.x1CoordEdit_273.text())
                self.solar_zenith_p3 =float(self.x1CoordEdit_264.text())
                self.solar_azimuth_p4 =float(self.x1CoordEdit_265.text())
                self.view_zenith_p5=float(self.x1CoordEdit_266.text())
                self.view_azimuth_p6= float(self.x1CoordEdit_267.text())
                self.month_p7=float(self.x1CoordEdit_268.text())
                self.day_p8=float(self.x1CoordEdit_269.text())
                self.sensor_altitude_p9=float(self.x1CoordEdit_270.text())
                self.target_alt_p10=float(self.x1CoordEdit_271.text())
                self.p1 = float(self.x1CoordEdit_274.text())
                self.p2 = float(self.x1CoordEdit_275.text())
                self.p3 = float(self.x1CoordEdit_276.text())
                if self.radioPixel_radiance.isChecked():
                    type_pixel='pixel_radiance'
                elif self.radioPixel_reflectance.isChecked():
                    type_pixel = 'pixel_reflectance'
                print(filepath)
                s = SixS(path=filepath)
                if not isinstance(s, SixS):
                    print('Cannot instantiate SixS')

        #------------- Atmospheric profile------------------------
                if self.checkBox_29.isChecked():
                    self.water= float(self.x1CoordEdit_262.text())
                    self.ozone = float(self.x1CoordEdit_263.text())
                    s.atmos_profile = AtmosProfile.UserWaterAndOzone(self.water,self.ozone)
                elif self.checkBox_30.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.NoGaseousAbsorption)
                elif self.checkBox_31.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)
                elif self.checkBox_32.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
                elif self.checkBox_33.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter)
                elif self.checkBox_34.isChecked():
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticSummer)
                else:
                    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticWinter)

        # ------------- Aero profile------------------------
                if self.checkBox_36.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
                elif self.checkBox_37.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Continental)
                elif self.checkBox_38.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Maritime)
                elif self.checkBox_39.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Urban)
                elif self.checkBox_40.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Desert)
                elif self.checkBox_41.isChecked():
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.BiomassBurning)
                else:
                    s.aeroprofile = AeroProfile.PredefinedType(AeroProfile.Stratospheric)

        # ------------- Run Simulation------------------------
                print('Simulation Running...')
                s.wavelength = Wavelength(self.Wavelength_p1)
                s.ground_reflectance = GroundReflectance.HomogeneousWalthall(self.p1,self.p2,self.p3,self.Reflectance_p2)
                    # HomogeneousWalthall(0.48, 0.50, 2.95, 0.6)#  0.0512 0.3169 -0.3085 0.065HomogeneousWalthall(0.0512, 0.3169, -0.3085, 0.065)#HomogeneousRoujean(0.37, 0.0, 0.133)#
                s.geometry.solar_z = self.solar_zenith_p3
                s.geometry.solar_a = self.solar_azimuth_p4
                s.geometry.view_z = self.view_zenith_p5
                s.geometry.view_a = self.view_azimuth_p6
                s.geometry.day = self.month_p7
                s.geometry.month = self.day_p8
                s.altitudes.set_sensor_satellite_level=self.sensor_altitude_p9
                s.altitudes.set_target_sea_level=self.target_alt_p10

                SixSHelpers.Angles.run_and_plot_360(s, 'view', output_name=type_pixel,colorbarlabel=str(type_pixel))
                plt.show()
                # SixSHelpers.Angles.run_and_plot_360(s, 'solar', output_name=type)
                # results, azimuths, zeniths, s.geometry.solar_a, s.geometry.solar_z = SixSHelpers.Angles.run360(s, 'view', output_name=type)
                # self.plot_polar_contour(results, azimuths, zeniths, filled=True)

                # data.to_csv('D:/test.csv')
                # self.mplWidgetSpectral_19.ax.plot(ax)
                # self.mplWidgetSpectral_19.ax.colorbar()

                # self.mplWidgetSpectral_19.canvas.draw()
                print('Simulation Completed Successfully...')
                QApplication.restoreOverrideCursor()
            except Exception as e:
                import traceback
                print(e, traceback.format_exc())
                QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    # QSizePolicy sretain=Form.sizePolicy()
    # sretain.setRetainSizeWhenHidden(True)
    # sretain.setSizePolicy()
    ui = Py6s()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())