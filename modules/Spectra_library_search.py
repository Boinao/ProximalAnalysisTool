# -*- coding: utf-8 -*-

from PyQt5 import QtCore
from PyQt5 import QtWidgets
import cv2 as cv

import specdal
from scipy.stats import kruskal
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from Ui.Spectra_Library_searchUi import Ui_Form
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
import spectral as spy
from spectral.database.usgs import USGSDatabase

# from . import GdalTools_utils as Utils
POSTFIX = '_Spectra_library_search'
from os import path
import seaborn as sns
import matplotlib.patches as mpatches

pluginPath = os.path.split(os.path.dirname(__file__))[0]


class SpectraLibrarySearch(Ui_Form):

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
        super(SpectraLibrarySearch, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.radioButton_2.toggled.connect(self.toggleoff_userinput)
        self.radioButton_3.toggled.connect(self.toggleoff_userinput)

        self.radioButton_5.toggled.connect(self.legend_state)
        self.radioButton_6.toggled.connect(self.legend_state)

    def toggleoff_userinput(self):
        # self.lineEdit.setDisabled(self.radioButton_3.isChecked())
        # self.lineEdit_2.setDisabled(self.radioButton_3.isChecked())

        if self.radioButton_2.isChecked():
            self.comboBox_classA_8.clear()
            self.comboBox_classA_8.addItem("--Select--")
            spectra_type = ['manmade', 'meteorites', 'mineral', 'non photosynthetic vegetation', 'rock',
                                 'soil', 'vegetation', 'water']

            for item in spectra_type:
                self.comboBox_classA_8.addItem(str(item))

            self.comboBox_classA_8.currentIndexChanged.connect(self.spectra_search_ecostress)

        if self.radioButton_3.isChecked():
            usgsspectra_type = ['ChapterS_SoilsAndMixtures', 'ChapterC_Coatings', 'ChapterL_Liquids',
                                'ChapterM_Minerals', 'ChapterO_OrganicCompounds', 'ChapterS_SoilsAndMixtures',
                                'ChapterV_Vegetation']
            self.comboBox_classA_8.clear()
            self.comboBox_classA_8.addItem("--Select--")
            for item in usgsspectra_type:
                self.comboBox_classA_8.addItem(str(item))

            self.comboBox_classA_8.currentIndexChanged.connect(self.spectra_search_usgs)


    def spectra_search_ecostress(self):
        try:
            self.widget_feature_5.clear()
            self.database = os.path.join(pluginPath, 'external/SpectralLibrary', 'ecostress.db')
            db = spy.EcostressDatabase(self.database)
            # min_wave = str(float(self.lineEdit.text()))
            # max_wave = str(float(self.lineEdit_2.text()))
            # spec_ID = int(self.lineEdit_3.text())
            label = str(self.comboBox_classA_8.currentText())
            # rows = db.query('SELECT SpectrumID FROM Samples, Spectra ' +
            #                 'WHERE Samples.SampleID = Spectra.SampleID AND ' +
            #                 'Type LIKE ' + "'" + label + "'" + ' AND' +
            #                 ' MinWavelength <= ' + min_wave + ' AND MaxWavelength >= ' + max_wave)

            # ids = [r[0] for r in rows]
            print("Spectral ID ", "Spectral Type")
            sql='SELECT SampleID, Name FROM Samples WHERE Type LIKE ' + "'" + label + "'"
            # db.print_query('SELECT SampleID, Name FROM Samples WHERE Type LIKE ' + "'" + label + "'")

            ret = db.query(sql)
            snames=[]
            for row in ret:
                snames.append(" - ".join([str(x) for x in row]))

            self.listWidget.clear()
            for sname in snames:
                self.listWidget.addItem(sname)
        except Exception as e:
            import traceback
            print(e, traceback.format_exc())



    def plot(self):
        self.widget_feature_5.clear()

        indexes = self.listWidget.selectedIndexes()
        specs = [index.row()+1 for index in indexes]
        legend = [item.text() for item in self.listWidget.selectedItems()]
        # print(legend)

        if self.radioButton_2.isChecked():
            db = spy.EcostressDatabase(self.database)
            i = 0
            for spec_ID in specs:
                s = db.get_signature(int(spec_ID))

                self.widget_feature_5.ax.plot(s.x, s.y, label=str(legend[i]))
                self.widget_feature_5.ax.set_xlabel("wavelength(micrometers)")
                self.widget_feature_5.ax.set_ylabel("Reflectance")
                self.widget_feature_5.ax.legend()
                # legend=self.widget_feature_5.ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
                # self.widget_feature_5.ax.add_artist(legend)
                self.widget_feature_5.canvas.draw()

                i += 1


        else:
            db = USGSDatabase(self.usgs_database)
            i = 0
            for spec_ID in specs:
                s = db.get_spectrum(int(spec_ID))
                self.widget_feature_5.ax.plot(s[0][1:-1], s[1][1:-1], label=str(legend[i]))
                self.widget_feature_5.ax.set_xlabel("wavelength(micrometers)")
                self.widget_feature_5.ax.set_ylabel("Reflectance")
                self.widget_feature_5.ax.legend()
                self.widget_feature_5.canvas.draw()
                i += 1
        self.legend_state()

    def getLegendSwitch(self):
        value = 0
        if self.radioButton_5.isChecked():
            value = 1
        else:
            value = 0
        return value

    def legend_state(self):
        if self.widget_feature_5.ax.get_lines():
            self.widget_feature_5.ax.legend().set_visible(self.getLegendSwitch())
            self.widget_feature_5.canvas.draw()




    def spectra_search_usgs(self):
        try:
            self.usgs_database = os.path.join(pluginPath, 'external/SpectralLibrary', 'usgs_lib.db')
            self.widget_feature_5.clear()
            db = USGSDatabase(self.usgs_database)
            # min_wave = str(float(self.lineEdit.text()))
            # max_wave = str(float(self.lineEdit_2.text()))
            # spec_ID = int(self.lineEdit_3.text())
            label = str(self.comboBox_classA_8.currentText())
            # rows = db.query('SELECT SampleID FROM Samples, Spectra ' +
            #                 'WHERE Samples.SampleID = Spectra.SampleID AND ' +
            #                 'Type LIKE ' + "'" + label + "'" + ' AND' +
            #                 ' MinValue <= ' + min_wave + ' AND MaxValue >= ' + max_wave)
            rows = db.query(
                'SELECT SampleID, FileName,Spectrometer FROM Samples WHERE Chapter LIKE ' + "'" + label + "'")

            ids = [r[0] for r in rows]
            # print("ID ", "Spectral Type")
            # db.print_query(
            #     'SELECT SampleID, FileName,Spectrometer FROM Samples WHERE Chapter LIKE ' + "'" + label + "'")
            sql='SELECT SampleID, FileName,Spectrometer FROM Samples WHERE Chapter LIKE ' + "'" + label + "'"
            ret = db.query(sql)
            snames = []
            for row in ret:
                snames.append(" - ".join([str(x) for x in row]))

            self.listWidget.clear()
            for sname in snames:
                self.listWidget.addItem(sname)
        except Exception as e:
            import traceback
            print(e, traceback.format_exc())



    def run(self):
        if ((not self.radioButton_2.isChecked()) and (not self.radioButton_3.isChecked())):
            self.radioButton_2.setFocus()
            messageDisplay = "Please select any of the given spectral library first"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if self.comboBox_classA_8.currentText()=='--Select--':
            self.comboBox_classA_8.setFocus()
            messageDisplay = "Please select target type first"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        indexes = self.listWidget.selectedIndexes()
        # print(indexes, len(indexes))
        if len(indexes)==0:
            self.listWidget.setFocus()
            messageDisplay = "Please select atleast one to show the spectra"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return


        try:
            self.database = os.path.join(pluginPath, 'external/SpectralLibrary', 'ecostress.db')
            self.usgs_database = os.path.join(pluginPath, 'external/SpectralLibrary',
                                              'usgs_lib.db')

            self.plot()
            # if self.radioButton_2.isChecked():
            #     self.plot()
            #     # self.spectra_search_ecostress()
            #     # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            #     # self.comboBox_classA_8.clear()
            #     # self.comboBox_classA_8.addItem("--Select--")
            #     # self.spectra_type = ['manmade', 'meteorites', 'mineral', 'non photosynthetic vegetation', 'rock',
            #     #                      'soil', 'vegetation', 'water']
            #     #
            #     # for item in self.spectra_type:
            #     #     self.comboBox_classA_8.addItem(str(item))
            #     # QApplication.restoreOverrideCursor()
            #     # self.comboBox_classA_8.currentIndexChanged.connect(self.spectra_search_ecostress)
            # if self.radioButton_3.isChecked():
            #     self.plot()
            #     # self.spectra_search_usgs()
            #     # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            #     # self.usgsspectra_type = ['ChapterS_SoilsAndMixtures','ChapterC_Coatings','ChapterL_Liquids','ChapterM_Minerals','ChapterO_OrganicCompounds','ChapterS_SoilsAndMixtures','ChapterV_Vegetation']
            #     # self.comboBox_classA_8.clear()
            #     # self.comboBox_classA_8.addItem("--Select--")
            #     # for item in self.usgsspectra_type:
            #     #     self.comboBox_classA_8.addItem(str(item))
            #     # QApplication.restoreOverrideCursor()
            #     # self.comboBox_classA_8.currentIndexChanged.connect(self.spectra_search_usgs)

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
    ui = SpectraLibrarySearch()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
