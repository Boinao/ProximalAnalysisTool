# -*- coding: utf-8 -*-

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtWidgets
import cv2 as cv
import re
import specdal
from scipy.stats import kruskal
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QCursor
from PyQt5.QtCore import Qt
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from Ui.Spectra_Library_matchUI import Ui_Form
import os
import collections
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
from scipy.spatial import distance
import matplotlib.pyplot as plt
import spectral as spy

# from . import GdalTools_utils as Utils
POSTFIX = '_Spectra_library_search'
from os import path
import seaborn as sns
import matplotlib.patches as mpatches
from numpy import genfromtxt

pluginPath = os.path.split(os.path.dirname(__file__))[0]

from modules import Utils

class SpectraLibraryMatch(Ui_Form):

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
        super(SpectraLibraryMatch, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.centralwavelength_clicked())
        self.radioButton_3.toggled.connect(self.toggleoff_userinput)
        self.toggleoff_userinput()

    def toggleoff_userinput(self):
        self.lineEdit.setDisabled(self.radioButton_3.isChecked())
        self.lineEdit_2.setDisabled(self.radioButton_3.isChecked())

        if self.radioButton_2.isChecked():
            self.comboBox.clear()
            self.comboBox.addItem("--Select--")
            spectra_type = ['manmade', 'meteorites', 'mineral', 'non photosynthetic vegetation', 'rock',
                                 'soil', 'vegetation', 'water']

            for item in spectra_type:
                self.comboBox.addItem(str(item))

            # self.comboBox_classA_8.currentIndexChanged.connect(self.spectra_search_ecostress)

        if self.radioButton_3.isChecked():
            usgsspectra_type = ['ChapterS_SoilsAndMixtures', 'ChapterC_Coatings', 'ChapterL_Liquids',
                                'ChapterM_Minerals', 'ChapterO_OrganicCompounds', 'ChapterS_SoilsAndMixtures',
                                'ChapterV_Vegetation']
            self.comboBox.clear()
            self.comboBox.addItem("--Select--")
            for item in usgsspectra_type:
                self.comboBox.addItem(str(item))

            # self.comboBox_classA_8.currentIndexChanged.connect(self.spectra_search_usgs)

    def SpectrabrowseButton_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_5.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            self.lineEdit_5.setText("")
            return

        self.spectra_filepath = fname

        # print(self.filepath)
        if fname:
            # inputText = str(fname) + " "
            self.lineEdit_5.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))



    def centralwavelength_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_4.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.hdr)", directory=lastDataDir)

        if not fname:
            self.lineEdit_4.setText("")
            return

        self.Central_wavelength_filepath = fname

        # print(self.filepath)
        if fname:
            # inputText = str(fname) + " "
            self.lineEdit_4.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))



    def spectra_match(self):
        self.widget_feature_5.ax.clear()
        db = spy.EcostressDatabase(self.database)
        min_wave = str(float(self.lineEdit.text()))
        max_wave = str(float(self.lineEdit_2.text()))
        label = str(self.comboBox.currentText())

        rows = db.query('SELECT SpectrumID FROM Samples, Spectra ' +
                        'WHERE Samples.SampleID = Spectra.SampleID AND ' +
                        'Type LIKE ' + "'" + label + "'" + ' AND' +
                        ' MinWavelength <= ' + min_wave + ' AND MaxWavelength >= ' + max_wave)
        # print(label,'SELECT SpectrumID FROM Samples, Spectra ' +
        #                 'WHERE Samples.SampleID = Spectra.SampleID AND ' +
        #                 'Type LIKE ' + "'" + label + "'" + ' AND' +
        #                 ' MinWavelength <= ' + min_wave + ' AND MaxWavelength >= ' + max_wave)

        ids = [r[0] for r in rows]

        # print(ids)
        if ids != []:
            if np.any(self.cw > 200):
                spy.BandInfo.centers = list(self.cw / 1000)
                spy.BandInfo.bandwidths = list(self.fw / 1000)
            lib = db.create_envi_spectral_library(ids, spy.BandInfo)

            # print('yes!!!')
            if self.radioButton_4.isChecked():
                dist = np.zeros((len(ids)))
                for i in range(0, len(ids)):
                    a = lib.spectra[i]
                    a[np.isnan(a)] = 0
                    b = self.my_data
                    b[np.isnan(b)] = 0
                    dist[i] = distance.euclidean(a, b)
                min_id = np.argsort(dist)
                q = str(ids[min_id[0]])
            elif self.radioButton_5.isChecked():
                dist = np.zeros((len(ids)))
                for i in range(0, len(ids)):
                    a = lib.spectra[i]
                    a[np.isnan(a)] = 0
                    b = self.my_data
                    b[np.isnan(b)] = 0
                    dist[i] = distance.correlation(a, b)
                min_id = np.argsort(dist)
                q = str(ids[min_id[-1]])
            elif self.radioButton_6.isChecked():
                dist = np.zeros((len(ids)))
                for i in range(0, len(ids)):
                    a = lib.spectra[i]
                    a[np.isnan(a)] = 0
                    b = self.my_data
                    b[np.isnan(b)] = 0
                    dist[i] = distance.cosine(a, b)
                min_id = np.argsort(dist)
                q = str(ids[min_id[0]])

            match = lib.spectra[min_id[0]] / np.max(lib.spectra[min_id[0]])
            name = db.query('SELECT Name FROM Samples WHERE Samples.SampleID =' + q)
            name_spec = [r[0] for r in name]
            self.widget_feature_5.ax.plot(self.cw, match, label=str(name_spec[0]))
            self.widget_feature_5.ax.plot(self.cw, b, label='input spectrum')
            self.widget_feature_5.ax.set_xlabel("Wavelength")
            self.widget_feature_5.ax.set_ylabel("Reflectance")
            self.widget_feature_5.ax.legend()
            self.widget_feature_5.canvas.draw()

            if self.radioButton_5.isChecked():
                for i in range(len(ids) - 6, len(ids)):
                    q = str(ids[min_id[i]])
                    name = db.query('SELECT Name FROM Samples WHERE Samples.SampleID =' + q)
                    name_spec = [r[0] for r in name]
                    sample_id = db.query('SELECT SampleID FROM Samples WHERE Samples.SampleID =' + q)
                    idname_spec = [r[0] for r in sample_id]
                    print('Sample Id=' + str(idname_spec[0]) + ';  Sample Name= ' + str(
                        name_spec[0]) + ';  Spectral Distance = ' + str(dist[min_id[i]]))
            else:
                for i in range(0, 5):
                    q = str(ids[min_id[i]])
                    name = db.query('SELECT Name FROM Samples WHERE Samples.SampleID =' + q)
                    name_spec = [r[0] for r in name]
                    sample_id = db.query('SELECT SampleID FROM Samples WHERE Samples.SampleID =' + q)
                    idname_spec = [r[0] for r in sample_id]
                    print('Sample Id=' + str(idname_spec[0]) + ';  Sample Name= ' + str(
                        name_spec[0]) + ';  Spectral Distance = ' + str(dist[min_id[i]]))
            # Save csv
            fname = self.lineEdit_5.text()
            path = os.path.dirname(fname)
            df1 = pd.DataFrame(match, index=self.cw)
            df1.to_csv(os.path.join(path,str(name_spec[0]) + '_Resampled_ECOSTRESS' + '.csv'), header=[str(name_spec[0])], index_label='Wavelength', index=True)
            print('Query Result saved in File :', os.path.join(path,str(name_spec[0]) + '_Resampled_ECOSTRESS' + '.csv') )

        else:
            messageDisplay = "Defined wavelength Range of the input spectrum is outside the range of the spectral library"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

    def spectra_match_usgs(self):
        self.widget_feature_5.ax.clear()
        db = spy.USGSDatabase(self.usgs_database)
        label = str(self.comboBox.currentText())
        # print(label)
        rows = db.query('SELECT SampleID, FileName,Spectrometer FROM Samples WHERE Chapter LIKE ' + "'" + label + "'")

        ids = [r[0] for r in rows]

        if ids!=[]:
            if np.any(self.cw > 200):
                spy.BandInfo.centers = list(self.cw / 1000)
                spy.BandInfo.bandwidths = list(self.fw / 1000)
            lib = db.create_envi_spectral_library(ids, spy.BandInfo)
            dist = np.zeros((len(ids)))
            if self.radioButton_4.isChecked():
                for i in range(0, len(ids)):
                    a = lib.spectra[i]
                    a[np.isnan(a)] = 0
                    b = self.my_data
                    b[np.isnan(b)] = 0
                    dist[i] = distance.euclidean(a[1:-1], b[1:-1])
                min_id = np.argsort(dist)
                q = str(ids[min_id[0]])
                match = lib.spectra[min_id[0]] / np.max(lib.spectra[min_id[0]])
                name = db.query('SELECT FileName TEXT FROM Samples WHERE Samples.SampleID =' + q)
                name_spec = [r[0] for r in name]
            elif self.radioButton_5.isChecked():
                for i in range(0, len(ids)):
                    a = lib.spectra[i]
                    a[np.isnan(a)] = 0
                    b = self.my_data
                    b[np.isnan(b)] = 0
                    dist[i] = distance.correlation(a[1:-1], b[1:-1])
                min_id = np.argsort(dist)
                q = str(ids[min_id[-1]])
                match = lib.spectra[min_id[-1]] / np.max(lib.spectra[min_id[-1]])
                name = db.query('SELECT FileName TEXT FROM Samples WHERE Samples.SampleID =' + q)
                name_spec = [r[0] for r in name]
            elif self.radioButton_6.isChecked():
                for i in range(0, len(ids)):
                    a = lib.spectra[i]
                    a[np.isnan(a)] = 0
                    b = self.my_data
                    b[np.isnan(b)] = 0
                    dist[i] = distance.cosine(a[1:-1], b[1:-1])
                min_id = np.argsort(dist)
                q = str(ids[min_id[0]])
                match = lib.spectra[min_id[0]] / np.max(lib.spectra[min_id[0]])
                name = db.query('SELECT FileName TEXT FROM Samples WHERE Samples.SampleID =' + q)
                name_spec = [r[0] for r in name]

            self.widget_feature_5.ax.plot(self.cw[1:-1], match[1:-1], label=str(name_spec[0]))
            self.widget_feature_5.ax.plot(self.cw[1:-1], b[1:-1], label='input spectrum')
            self.widget_feature_5.ax.set_ylim(0,1.5)
            self.widget_feature_5.ax.set_xlabel("Wavelength")
            self.widget_feature_5.ax.set_ylabel("Reflectance")
            self.widget_feature_5.ax.legend()
            self.widget_feature_5.canvas.draw()

            if self.radioButton_5.isChecked():
                for i in range(len(ids)-6, len(ids)):
                    q = str(ids[min_id[i]])
                    match = lib.spectra[min_id[i]] / np.max(lib.spectra[min_id[i]])
                    name = db.query('SELECT FileName TEXT FROM Samples WHERE Samples.SampleID =' + q)
                    name_spec = [r[0] for r in name]
                    id = db.query('SELECT SampleID FROM Samples WHERE Samples.SampleID =' + q)
                    idname_spec = [r[0] for r in id]
                    print('Sample Id=' + str(idname_spec[0]) + ';  Sample Name= ' + str(
                        name_spec[0]) + ';  Spectral Distance = ' + str(dist[min_id[i]]))
            else:
                for i in range(0, 5):
                    q = str(ids[min_id[i]])
                    match = lib.spectra[min_id[i]] / np.max(lib.spectra[min_id[i]])
                    name = db.query('SELECT FileName TEXT FROM Samples WHERE Samples.SampleID =' + q)
                    name_spec = [r[0] for r in name]
                    id = db.query('SELECT SampleID FROM Samples WHERE Samples.SampleID =' + q)
                    idname_spec = [r[0] for r in id]
                    print('Sample Id=' + str(idname_spec[0]) + ';  Sample Name= ' + str(
                        name_spec[0]) + ';  Spectral Distance = ' + str(dist[min_id[i]]))

            # Save csv
            fname=self.lineEdit_5.text()
            path=os.path.dirname(fname)
            df1 = pd.DataFrame(match, index=self.cw)
            df1.to_csv(os.path.join(path,str(name_spec[0]) + '_Resampled_USGS' + '.csv'), header=[str(name_spec[0])], index_label='Wavelength', index=True)
            print('Query Result saved in File :', os.path.join(path,str(name_spec[0]) + '_Resampled_USGS' + '.csv'))

        else:
            messageDisplay = "Defined wavelength Range of the input spectrum is outside the range of the spectral library"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

    def run(self):

        if (self.lineEdit_5.text() is None) or (self.lineEdit_5.text() == ""):
            self.lineEdit_5.setFocus()
            messageDisplay = "Cannot leave field empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_4.text() is None) or (self.lineEdit_4.text() == ""):
            self.lineEdit_4.setFocus()
            messageDisplay = "Cannot leave field empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_5.text() )):
            self.lineEdit_5.setFocus()
            messageDisplay = "Input Path does not exist "
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_4.text() )):
            self.lineEdit_4.setFocus()
            messageDisplay = "Input Path does not exist "
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if self.radioButton_2.isChecked():
            if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
                self.lineEdit.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if (self.lineEdit_2.text() is None) or (self.lineEdit_2.text() == ""):
                self.lineEdit_2.setFocus()
                messageDisplay = "Cannot leave field empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        if self.comboBox.currentText()=='--Select--':
            self.comboBox.setFocus()
            messageDisplay = "Please select target type first"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return




        try:
            # Spectra
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            my_data = genfromtxt(self.spectra_filepath, delimiter=',')[1::, 1]
            self.my_data = my_data / np.max(my_data)
            self.database = os.path.join(pluginPath, 'external','SpectralLibrary', 'ecostress.db')
            self.usgs_database = os.path.join(pluginPath, 'external','SpectralLibrary',
                                              'usgs_lib.db')
            # HDR file read
            hdr = Utils.read_hdr_file(self.Central_wavelength_filepath, keep_case=False)
            fw1 = str.split(hdr['fwhm'], ',')
            fw = []
            for i in range(0, len(fw1)):
                c = str.lstrip(fw1[i])
                c = float(c)
                c = float("{0:.2f}".format(c))
                fw = np.append(fw, c)
            self.fw = fw
            # hdr=read_hdr_file(r'F:/SOIL HYPERSPECTRAL/chilika/roi/ang20151226t043231_corr_v2m2_img.hdr',keep_case=False)
            cw1 = str.split(hdr['wavelength'], ',')
            cw = []
            for i in range(0, len(cw1)):
                c = str.lstrip(cw1[i])
                c = float(c)
                c = float("{0:.3f}".format(c))
                cw = np.append(cw, c)
            self.cw = cw

            if self.radioButton_2.isChecked():
                self.spectra_match()
                # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.spectra_type = ['manmade', 'meteorites', 'mineral', 'non photosynthetic vegetation',
                #                      'rock', 'soil', 'vegetation', 'water']
                # self.comboBox.clear()
                # self.comboBox.addItem("--Select--")
                # for item in self.spectra_type:
                #     self.comboBox.addItem(str(item))
                # QApplication.restoreOverrideCursor()
                # self.comboBox.currentIndexChanged.connect(self.spectra_match)
            if self.radioButton_3.isChecked():
                self.spectra_match_usgs()
                # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.usgsspectra_type = ['ChapterS_SoilsAndMixtures', 'ChapterC_Coatings', 'ChapterL_Liquids',
                #                          'ChapterM_Minerals', 'ChapterO_OrganicCompounds', 'ChapterS_SoilsAndMixtures',
                #                          'ChapterV_Vegetation']
                # self.comboBox.clear()
                # self.comboBox.addItem("--Select--")
                # for item in self.usgsspectra_type:
                #     self.comboBox.addItem(item)
                # QApplication.restoreOverrideCursor()
                # self.comboBox.currentIndexChanged.connect(self.spectra_match_usgs)
            QApplication.restoreOverrideCursor()
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
    ui = SpectraLibraryMatch()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
