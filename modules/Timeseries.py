# -*- coding: utf-8 -*-
"""
Created on Mon January 4 11:05:06 2021

@author: Nidhin
"""
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import cv2 as cv
import specdal
from scipy.stats import kruskal
# from PyQt5.QtCore import *

# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QDialog
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QCursor
from Ui.TimeseriesUi import Ui_Form
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
POSTFIX = '_Timeseries'
from os import path
import seaborn as sns
import matplotlib.patches as mpatches

from modules import Utils



from PyQt5 import uic
pluginPath = os.path.split(os.path.dirname(__file__))[0]
# print(pluginPath)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'Ui', 'dialogResultViewer.ui'))

# from Ui.dialogResultViewer import Ui_Dialog

class Dialog(BASE, WIDGET):
    def __init__(self, title='Result Viewer', parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.setupUi(self)


class Timeseries(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.outputFilename = ""
        self.data = pd.DataFrame()

        self.metafilepath = None
        self.compareTwoClassesLoaded = False
        self.compareBandSepLoaded = False

    def get_widget(self):
        return self.groupBox

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def setupUi(self, Form):
        super(Timeseries, self).setupUi(Form)
        self.Form = Form
        self.x1CoordEdit_11.setText("0.0")
        self.label_73.setVisible(False)
        self.x1CoordEdit_11.setVisible(False)

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_5.clicked.connect(lambda: self.MetadatabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveasButton_clicked())
        self.rb_sep_compare_5.toggled.connect(self.toggleRadio_rb_sep_compare)
        self.radioClassDistribute.toggled.connect(self.toggleRadio_ClassDistribute)
        self.radioFTEST_5.toggled.connect(self.toggleRadio_FTEST)
        self.radioSEP_5.toggled.connect(self.toggleRadio_SEP)
        self.radioMUTUAL_5.toggled.connect(self.toggleRadio_Tukey)
        self.radioTREE_5.toggled.connect(self.toggleRadio_TREE)


        self.radioFeatureVar.toggled.connect(self.toggleRadio_radioFeatureVar)

        # self.radioPairplot_5.toggled.connect(self.toggleRadio_pairplot)
        # self.radioButton_feature.toggled.connect(self.toggleRadio_featureselect)
        # self.radioButton_bandnorm_5.toggled.connect(self.toggleRadio_bandnorm)
        # self.radioButton_bandwise_5.toggled.connect(self.toggleRadio_bandwise)

        # self.x1CoordEdit_11.setValidator(QIntValidator())
        self.x1CoordEdit_10.setValidator(QIntValidator())
        self.x1CoordEdit_13.setValidator(QIntValidator())

        self.tabWidget_4.setCurrentIndex(0)

    def toggleRadio_rb_sep_compare(self):
        self.comboBox_25.setEnabled(self.rb_sep_compare_5.isChecked())
        self.comboBox_26.setEnabled(self.rb_sep_compare_5.isChecked())

        self.toggleComboboxes()
        # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # self.comboBox_25.clear()
        # self.comboBox_25.addItem("--Select--")
        # # for i in np.unique(self.day):
        # #     self.comboBox_25.addItem("Day " + str(i))
        # # self.roi_ds = np.unique(self.day)
        #
        # # for i in range(0, self.no_days):
        # #     self.comboBox_25.addItem("class " + str(i))
        # # self.roi_ds = range(0, self.no_days)
        # for i in self.df.columns:
        #     self.comboBox_25.addItem("Day " + i)
        # self.roi_ds = self.df.columns
        #
        # QApplication.restoreOverrideCursor()
        # self.comboBox_25.currentIndexChanged.connect(self.updateSecondCombo)
        # self.comboBox_26.currentIndexChanged.connect(self.twoClassPlot)

    def toggleRadio_ClassDistribute(self):
        self.x1CoordEdit_11.setEnabled(self.radioClassDistribute.isChecked())




    def toggleRadio_FRA(self):
        self.comboBox_28.setEnabled(self.radioFRA_5.isChecked())

    def toggleRadio_SAM(self):
        self.comboBox_28.setEnabled(self.radioSAM_5.isChecked())

    def toggleRadio_FTEST(self):
        self.comboBox_classA_8.setEnabled(self.radioFTEST_5.isChecked())
        self.comboBox_classB_8.setEnabled(self.radioFTEST_5.isChecked())

        if self.radioFTEST_5.isChecked() or self.radioSEP_5.isChecked() or self.radioTREE_5.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not self.compareBandSepLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                classes=metadata.loc['Days'].unique().astype(int)


                self.comboBox_classA_8.clear()
                self.comboBox_classB_8.clear()
                self.comboBox_classA_8.addItem("--Select--")
                for i in classes:
                    self.comboBox_classA_8.addItem("Day " + str(i))

                QApplication.restoreOverrideCursor()
                self.SDIsecond=classes
                self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                self.compareBandSepLoaded=True
            # self.comboBox_26.currentIndexChanged.connect(self.twoClassPlot)

        # self.read_data()
        #
        # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # self.comboBox_classA_8.clear()
        # self.comboBox_classA_8.addItem("--Select--")
        # # for i in range(0, self.no_days):
        # #     self.comboBox_classA_8.addItem("class " + str(i))
        # # self.SDIsecond = range(0, self.no_days)
        # for i in self.df.columns:
        #     self.comboBox_classA_8.addItem("Day " + i)
        # self.SDIsecond = self.df.columns
        # QApplication.restoreOverrideCursor()
        # self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
        # self.comboBox_classB_8.currentIndexChanged.connect(self.Plot_Feature_Select)

    def toggleRadio_SEP(self):
        self.comboBox_classA_8.setEnabled(self.radioSEP_5.isChecked())
        self.comboBox_classB_8.setEnabled(self.radioSEP_5.isChecked())

        if self.radioFTEST_5.isChecked() or self.radioSEP_5.isChecked() or self.radioTREE_5.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not self.compareBandSepLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                classes=metadata.loc['Days'].unique().astype(int)


                self.comboBox_classA_8.clear()
                self.comboBox_classB_8.clear()
                self.comboBox_classA_8.addItem("--Select--")
                for i in classes:
                    self.comboBox_classA_8.addItem("Day " + str(i))

                QApplication.restoreOverrideCursor()
                self.SDIsecond=classes
                self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                self.compareBandSepLoaded=True
        # self.read_data()
        # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # self.comboBox_classA_8.clear()
        # self.comboBox_classA_8.addItem("--Select")
        # # for i in range(0, self.no_days):
        # #     self.comboBox_classA_8.addItem("class " + str(i))
        # # self.SDIsecond = range(0, self.no_days)
        # for i in self.df.columns:
        #     self.comboBox_classA_8.addItem("Day " + i)
        # QApplication.restoreOverrideCursor()
        # self.SDIsecond = self.df.columns
        # self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
        # self.comboBox_classB_8.currentIndexChanged.connect(self.OneWayAnova)

    def toggleRadio_Tukey(self):
        self.x1CoordEdit_10.setEnabled(self.radioMUTUAL_5.isChecked())

    def toggleRadio_TREE(self):
        self.comboBox_classA_8.setEnabled(self.radioTREE_5.isChecked())
        self.comboBox_classB_8.setEnabled(self.radioTREE_5.isChecked())

        if self.radioFTEST_5.isChecked() or self.radioSEP_5.isChecked() or self.radioTREE_5.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not self.compareBandSepLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                classes=metadata.loc[' '].unique().astype(int)


                self.comboBox_classA_8.clear()
                self.comboBox_classB_8.clear()
                self.comboBox_classA_8.addItem("--Select--")
                for i in classes:
                    self.comboBox_classA_8.addItem("Day " + str(i))

                QApplication.restoreOverrideCursor()
                self.SDIsecond=classes
                self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                self.compareBandSepLoaded=True



        # self.read_data()
        # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # self.comboBox_classA_8.clear()
        # self.comboBox_classA_8.addItem("--Select")
        # # for i in range(0, self.no_days):
        # #     self.comboBox_classA_8.addItem("class " + str(i))
        # # self.SDIsecond = range(0, self.no_days)
        # for i in self.df.columns:
        #     self.comboBox_classA_8.addItem("Day " + i)
        #
        # QApplication.restoreOverrideCursor()
        # self.SDIsecond = self.df.columns
        # self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
        # self.comboBox_classB_8.currentIndexChanged.connect(self.Kruskal_Wallis)


    # def toggleRadio_pairplot(self):
    #     self.comboBox_29.setEnabled(self.radioPairplot_5.isChecked())
    #     self.comboBox_30.setEnabled(self.radioPairplot_5.isChecked())

    # def toggleRadio_featureselect(self):
    #     self.comboBox_classA.setEnabled(self.radioButton_feature.isChecked())
    #     self.comboBox_classB.setEnabled(self.radioButton_feature.isChecked())

    def toggleRadio_radioFeatureVar(self):
        self.x1CoordEdit_13.setEnabled(self.radioFeatureVar.isChecked())
        # self.comboBox_3.setEnabled(self.radioFeatureVar.isChecked())

    def toggleRadio_TSNE(self):
        self.x1CoordEdit_9.setEnabled(self.radioButton_TSNE_5.isChecked())

    # def toggleRadio_bandnorm(self):
    #     # self.x1CoordEdit_10.setEnabled(self.radioButton_bandnorm_5.isChecked())
    #     self.comboBox_classA_9.setEnabled(self.radioButton_bandnorm_5.isChecked())
    #     self.comboBox_classB_9.setEnabled(self.radioButton_bandnorm_5.isChecked())

    # def toggleRadio_bandwise(self):
    #     # self.x1CoordEdit_2.setEnabled(self.radioButton_bandwise.isChecked())
    #     self.comboBox_classA_9.setEnabled(self.radioButton_bandwise_5.isChecked())
    #     self.comboBox_classB_9.setEnabled(self.radioButton_bandwise_5.isChecked())
    def read_data(self):

        if (self.lineEdit_2.text() is None) or (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_2.text())):
            self.lineEdit_2.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                           "Kindly enter a valid input file.",
                                           QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
            self.lineEdit.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit.text())):
            self.lineEdit.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid", "Kindly enter a valid file.",
                                           QtWidgets.QMessageBox.Ok)
            return

        # if (self.lineEdit_3.text() is None) or (self.lineEdit_3.text() == ""):
        #     self.lineEdit_3.setFocus()
        #     QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Output File is required",
        #                                   QtWidgets.QMessageBox.Ok)
        #     return
        #
        # if (not os.path.isdir(os.path.dirname(self.lineEdit_3.text()))):
        #     self.lineEdit_3.setFocus()
        #     QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
        #                                    "Kindly enter a valid output path.",
        #                                    QtWidgets.QMessageBox.Ok)
        #     return

        try:
            self.inFile = self.lineEdit_2.text()
            # self.outFile = self.lineEdit_3.text()
            self.Metafile = self.lineEdit.text()

            # Read in our image and ROI image
            # img_ds = gdal.Open(inFile, gdal.GA_ReadOnly)
            # roi_ds = gdal.Open(self.Metafile, gdal.GA_ReadOnly)
            df_spectra = pd.read_csv(self.filepath,
                                     header=0, index_col=0)
            self.spectra = df_spectra.to_numpy()
            roi_ds = pd.read_csv(self.metafilepath, header=None, index_col=0)
            # for i in range(1, len(roi_ds)):
            #     roi = roi_ds.iloc[i, 0]
            #     print(roi, 'roi')
            #     self.roi = roi
            df = roi_ds.loc['Days']
            dupl = df.duplicated()
            self.wavelength = df_spectra.index.values

            min = np.min(self.wavelength)
            max = np.max(self.wavelength)

            dupl_obs_samples = dupl[dupl == False].index.values
            self.no_days = np.shape(dupl_obs_samples)[0]
            self.days_emerg = roi_ds.loc['Days'].values
            dupl_obs_samples = np.append(dupl_obs_samples, self.spectra.shape[1] + 1)
            self.dupl_obs_samples = dupl_obs_samples - 1
            self.day = np.unique(self.days_emerg).astype(int)
            self.df_mean = pd.DataFrame()
            self.df_std = pd.DataFrame()
            self.df = pd.DataFrame()
            for i in range(0, self.no_days):
                self.df_mean[str(int(self.days_emerg[dupl_obs_samples[i]]))] = self.spectra[:,
                                                                               self.dupl_obs_samples[i]:
                                                                               self.dupl_obs_samples[
                                                                                   i + 1]].mean(axis=1)
                self.df_std[str(int(self.days_emerg[dupl_obs_samples[i]]))] = self.spectra[:,
                                                                              self.dupl_obs_samples[i]:
                                                                              self.dupl_obs_samples[
                                                                                  i + 1]].std(axis=1)
                self.df[str(int(self.days_emerg[dupl_obs_samples[i]]))] = self.spectra[:,
                                                                          self.dupl_obs_samples[i]:
                                                                          self.dupl_obs_samples[
                                                                              i + 1]].tolist()

        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",traceback.format_exc(),
                                           QtWidgets.QMessageBox.Ok)
            print(e, traceback.format_exc())
            return


    def flushFlag(self):

        self.compareTwoClassesLoaded = False
        self.compareBandSepLoaded = False

    # def onLayersChanged(self):
    #     self.inSelector.setLayers(Utils.LayerRegistry.instance().getRasterLayers())
    def toggleComboboxes(self):
        self.comboBox_25.setEnabled(self.rb_sep_compare_5.isChecked())
        self.comboBox_26.setEnabled(self.rb_sep_compare_5.isChecked())

        if self.rb_sep_compare_5.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not self.compareTwoClassesLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                classes=metadata.loc['Days'].unique().astype(int)


                self.comboBox_25.clear()
                self.comboBox_26.clear()
                self.comboBox_25.addItem("--Select--")
                for i in classes:
                    self.comboBox_25.addItem("Day " + str(i))

                QApplication.restoreOverrideCursor()
                self.roi_ds=classes
                self.comboBox_25.currentIndexChanged.connect(self.updateSecondCombo)
                self.compareTwoClassesLoaded=True
            # self.comboBox_26.currentIndexChanged.connect(self.twoClassPlot)


        # self.read_data()
        # # self.toggleComboboxes()
        # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # self.comboBox_25.clear()
        # self.comboBox_25.addItem("--Select--")
        # # for i in np.unique(self.day):
        # #     self.comboBox_25.addItem("Day " + str(i))
        # # self.roi_ds = np.unique(self.day)
        #
        # # for i in range(0, self.no_days):
        # #     self.comboBox_25.addItem("class " + str(i))
        # # self.roi_ds = range(0, self.no_days)
        # for i in self.df.columns:
        #     self.comboBox_25.addItem("Day " + i)
        # self.roi_ds = self.df.columns
        #
        # QApplication.restoreOverrideCursor()
        # self.comboBox_25.currentIndexChanged.connect(self.updateSecondCombo)
        # self.comboBox_26.currentIndexChanged.connect(self.twoClassPlot)

    def stateChanged1(self):
        self.lineEdit.setEnabled(True)

    def stateChanged2(self):
        self.lineEdit.setEnabled(True)

    def SpectrabrowseButton_clicked(self):
        self.flushFlag()
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_2.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            return

        self.filepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit_2.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))

            self.outputFilename = (os.path.dirname(fname)) + "/Output" + POSTFIX
            self.lineEdit_3.setText(self.outputFilename)
        else:
            self.lineEdit_2.setText("")

    def MetadatabrowseButton_clicked(self):
        self.flushFlag()
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            return

        self.metafilepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))

            self.outputFilename = (os.path.dirname(fname)) + "/Output" + POSTFIX
            self.lineEdit_3.setText(self.outputFilename)


        else:
            self.lineEdit.setText("")

    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')
        if not self.outputFilename:
            return

        self.lineEdit_3.setText(self.outputFilename)

        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename


    # MEAN PLOT
    # def plotMeanStd(self):
    #     self.mplWidgetSpectral_5.ax.clear()
    #     df_spectra = pd.read_csv(self.filepath, header=0, index_col=0)
    #     df_metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
    #     spectra = df_spectra.to_numpy()
    #
    #     wavelength = df_spectra.index.values
    #     days_emerg = df_metadata.loc['Days'].values
    #     # print(df_metadata, "metadata")
    #
    #     # //Compute Mean Spectra Based on the No. of Replications
    #     df = df_metadata.loc['Days']
    #     dupl = df.duplicated()
    #     dupl_obs_samples = dupl[dupl == False].index.values
    #
    #     no_days = np.shape(dupl_obs_samples)[0]
    #     avg_spectra = np.zeros((wavelength.shape[0], no_days))
    #
    #     dupl_obs_samples = np.append(dupl_obs_samples, spectra.shape[1])
    #     dupl_obs_samples = dupl_obs_samples - 1
    #
    #     for i in range(0, no_days):
    #         avg_spectra[:, i] = spectra[:, dupl_obs_samples[i]:dupl_obs_samples[i + 1]].mean(axis=1)
    #         self.mplWidgetSpectral_5.ax.plot(wavelength, avg_spectra[:, i],
    #                                          label='Day = ' + str(int(days_emerg[dupl_obs_samples[i]])),
    #                                          linewidth=3)
    #         self.mplWidgetSpectral_5.ax.legend()
    #         self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
    #         self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
    #     self.mplWidgetSpectral_5.canvas.draw()
    def clearButtonGrp(self):
        self.buttonGroup.setExclusive(False)
        self.buttonGroup.checkedButton().setChecked(False)
        self.buttonGroup.setExclusive(True)


    def plotCV(self):
        try:
            self.mplWidgetSpectral_5.ax.clear()
            df_spectra = pd.read_csv(self.filepath, header=0, index_col=0)
            df_metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)

            # print(df_spectra)

            no_sample_avg = 10
            spectra = df_spectra.to_numpy()
            wavelength = df_spectra.index.values
            crop_param = df_metadata.index
            spectra_id = df_metadata.loc['Spectra ID'].values
            samples = df_metadata.loc['Spectra ID'].values

            df_temp=df_metadata.drop(["Spectra ID", "class_label", "class"], axis=0)
            metadata = df_temp.to_numpy().astype('float')
            days_emerg = df_metadata.loc['Days'].values
            # print(df_metadata, "metadata")

            # //Compute Mean Spectra Based on the No. of Replications
            df = df_metadata.loc['Days']
            dupl = df.duplicated()
            dupl_obs_samples = dupl[dupl == False].index.values

            no_days = np.shape(dupl_obs_samples)[0]
            avg_spectra = np.zeros((wavelength.shape[0], no_days))

            dupl_obs_samples = np.append(dupl_obs_samples, spectra.shape[1])
            dupl_obs_samples = dupl_obs_samples - 1
            df = df_metadata.loc['Days']
            dupl = df.duplicated()
            dupl_obs_samples = dupl[dupl == False].index.values

            no_days = np.shape(dupl_obs_samples)[0]
            avg_param = np.zeros((metadata.shape[0], no_days))

            dupl_obs_samples = np.append(dupl_obs_samples, metadata.shape[1])
            dupl_obs_samples = dupl_obs_samples - 1
            for i in range(0, no_days):
                # print(dupl_obs_samples[i],dupl_obs_samples[i+1])
                # print(metadata[:, dupl_obs_samples[i]:dupl_obs_samples[i + 1]])
                avg_param[:, i] = metadata[:, dupl_obs_samples[i]:dupl_obs_samples[i + 1]].mean(axis=1)
                # a=spectra[:,dupl_obs_samples[i]:dupl_obs_samples[i+1]]
                # print(a.shape)
                # print(dupl_obs_samples[i])#df.duplicated().to_string())

            # //Parameter Summary
            # import researchpy as rp
            df = df_metadata.T
            # rp.summary_cont(df['DW (g/m2)'])
            df[["DW (g/m2)", "LAI", 'N (%)', 'WC (%)']].describe()

            # //Visulaize the Spectra using Fill Curve
            df = df_metadata.loc['Days']
            dupl = df.duplicated()
            dupl_obs_samples = dupl[dupl == False].index.values

            no_days = np.shape(dupl_obs_samples)[0]


            days_emerg = df_metadata.loc['Days'].values
            dupl_obs_samples = np.append(dupl_obs_samples, spectra.shape[1] + 1)
            dupl_obs_samples = dupl_obs_samples - 1
            avg_spectra_ = np.zeros((wavelength.shape[0], no_days))
            std_spectra_=np.zeros((wavelength.shape[0],no_days))

            for i in range(0, no_days):
                avg_spectra = spectra[:, dupl_obs_samples[i]:dupl_obs_samples[i + 1]].mean(axis=1)
                avg_spectra_[:, i]=avg_spectra
                std_spectra = spectra[:, dupl_obs_samples[i]:dupl_obs_samples[i + 1]].std(axis=1)
                std_spectra_[:, i]=std_spectra
                lower = avg_spectra - std_spectra
                upper = avg_spectra + std_spectra
                self.mplWidgetSpectral_5.ax.plot(wavelength, avg_spectra,
                                                 label='Day = ' + str(int(days_emerg[dupl_obs_samples[i]])), linewidth=3)
                # self.mplWidgetSpectral_5.ax.legend()
                legend = self.mplWidgetSpectral_5.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                legend.set_draggable(True)
                self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
                self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
                self.mplWidgetSpectral_5.ax.fill_between(self.wavelength, lower, upper, alpha=0.5)

            self.mplWidgetSpectral_5.canvas.draw()
            # print(avg_spectra_.shape,std_spectra_.shape)
            result=np.hstack((avg_spectra_,std_spectra_))
            # print(result.shape)
            df1 = pd.DataFrame(result, index=wavelength)
            # print([str(int(days_emerg[dupl_obs_samples[i]])) for i in range(no_days)]*2)
            head=['Day_Mean_'+str(int(days_emerg[dupl_obs_samples[i]])) for i in range(no_days)]+['Day_Std_'+str(int(days_emerg[dupl_obs_samples[i]])) for i in range(no_days)]
            df1.to_csv(self.outputFilename + '_Spectra' + '.csv', header=head, index=True)
        except Exception as e:
            import traceback
            print(e, traceback.format_exc())

    def twoClassPlot(self, i):
        self.mplWidgetSpectral_5.ax.clear()
        # df_spectra = pd.read_csv(self.filepath, header=0, index_col=0)
        # df_metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
        #
        # # print(df_spectra)
        #
        # no_sample_avg = 10
        # spectra = df_spectra.to_numpy()
        # wavelength = df_spectra.index.values
        # if i < 1:
        #     return
        # label_one = int(self.comboBox_25.currentText().split()[1])
        # label_two = int(self.comboBox_26.currentText().split()[1])
        #
        # ignored_labels = [0]
        # i = 0
        # df = df_metadata.loc['Days']
        # dupl = df.duplicated()
        # dupl_obs_samples = dupl[dupl == False].index.values
        # no_days = np.shape(dupl_obs_samples)[0]
        # avg_spectra = np.zeros((wavelength.shape[0], no_days))
        # days_emerg = df_metadata.loc['Days'].values
        #
        # # MASK
        # day = np.unique(days_emerg).astype(int)
        # mask = np.resize(day, (10, 1))
        # array1 = dupl_obs_samples
        # # x = np.where(array1 == array1, mask, array1)
        # # print(x, 'mma')
        #
        # dupl_day = np.append(day, spectra.shape[1])
        # dupl_obs_samples = np.append(dupl_obs_samples, spectra.shape[1])
        # dupl_obs_samples = dupl_obs_samples - 1
        #
        # spectra1 = spectra[:, dupl_obs_samples[label_one]:dupl_obs_samples[label_one + 1]].mean(axis=1)
        # print(spectra1, 'spectra1')
        # spectra2 = spectra[:, dupl_obs_samples[label_two]:dupl_obs_samples[label_two + 1]].mean(axis=1)
        #
        # self.mplWidgetSpectral_5.ax.plot(wavelength, spectra1,
        #                                  label='Day = ' + str(int(days_emerg[dupl_obs_samples[label_one]])),
        #                                  linewidth=3)
        # self.mplWidgetSpectral_5.ax.plot(wavelength, spectra2,
        #                                  label='Day = ' + str(int(days_emerg[dupl_obs_samples[label_two]])),
        #                                  linewidth=3)
        # self.mplWidgetSpectral_5.ax.legend()
        # self.mplWidgetSpectral_5.canvas.draw()
        print(i)
        if i < 1:
            return
        label_one = str(int(self.comboBox_25.currentText().split()[1]))
        label_two = str(int(self.comboBox_26.currentText().split()[1]))
        spectra1 = self.df_mean[label_one]

        spectra2 = self.df_mean[label_two]
        self.mplWidgetSpectral_5.ax.plot(self.wavelength, spectra1,
                                         label='Day = ' + str(label_one),
                                         linewidth=3)
        self.mplWidgetSpectral_5.ax.plot(self.wavelength, spectra2,
                                         label='Day = ' + str(label_two),
                                         linewidth=3)
        self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
        self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
        # self.mplWidgetSpectral_5.ax.legend()
        legend = self.mplWidgetSpectral_5.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        legend.set_draggable(True)
        self.mplWidgetSpectral_5.canvas.draw()

    def Plot_Continum_removal(self):
        self.mplWidgetSpectral_5.ax.clear()
        baseline = float(self.x1CoordEdit_11.text())
        for i in range(0, self.no_days):
            spectra1 = self.spectra[:, self.dupl_obs_samples[i]:self.dupl_obs_samples[i + 1]].mean(axis=1).tolist()
            wvl = self.wavelength.tolist()

            fea = spectro.FeaturesConvexHullQuotient(spectra1, wvl, baseline=baseline)
            self.mplWidgetSpectral_5.ax.plot(self.wavelength, fea.crs,label='Day = ' + str(int(self.days_emerg[self.dupl_obs_samples[i]])))
            self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
            self.mplWidgetSpectral_5.ax.set_ylabel('CR Values')
            # self.mplWidgetSpectral_5.ax.legend()
            legend = self.mplWidgetSpectral_5.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            legend.set_draggable(True)
        self.mplWidgetSpectral_5.canvas.draw()


    def fract(self,p_vec, q_vec):
        diff = p_vec - q_vec
        diff_fraction = diff ** 2
        return math.pow(np.sum(diff_fraction), 1 / 2)

    def updateSecondCombo(self, i):
        labels = self.roi_ds
        # labels = range(0, labels)
        second_labels = [x for x in labels if x != labels[i - 1]]
        # second_labels = [x for x in labels]
        self.mplWidgetSpectral_5.ax.clear()
        self.comboBox_26.clear()
        self.comboBox_26.addItem("--Select--")
        for item in second_labels:
            self.comboBox_26.addItem("Day " + str(item))


    def Band_plot(self):
        # self.mplWidgetSpectral_6.ax.clear()
        # test_wave = int(self.x1CoordEdit_13.text())
        # if i < 1:
        #     return
        # label_one = str(int(self.comboBox_3.currentText().split()[1]))
        # df = pd.DataFrame()
        # index = self.wavelength
        # for i in range(0, self.no_days):
        #     spectra1 = self.spectra[:, self.dupl_obs_samples[i]:self.dupl_obs_samples[i + 1]]
        #     spectra1_dataframe = pd.DataFrame(spectra1, index=index)
        #     df['day' + str(int(self.days_emerg[self.dupl_obs_samples[i]]))] = spectra1_dataframe.loc[test_wave]
        # self.mplWidgetSpectral_6.ax.plot(df['day' + label_one],label='Day = ' + str(label_one))
        #
        # self.mplWidgetSpectral_6.ax.plot(df,label=df.columns.name)
        # self.mplWidgetSpectral_6.ax.set_xlabel('no of samples')
        # self.mplWidgetSpectral_6.ax.set_ylabel('Values')
        # self.mplWidgetSpectral_6.ax.legend()
        # self.mplWidgetSpectral_6.canvas.draw()
        self.mplWidgetSpectral_6.ax.clear()
        test_wave = int(self.x1CoordEdit_13.text())
        # if i < 1:
        #     return
        # label_one = str(int(self.comboBox_3.currentText().split()[1]))
        df = pd.DataFrame()
        index = self.wavelength

        for i in range(0, self.no_days):
            spectra1 = self.spectra[:, self.dupl_obs_samples[i]:self.dupl_obs_samples[i + 1]]
            spectra1_dataframe = pd.DataFrame(spectra1, index=index)
            df['day' + str(int(self.days_emerg[self.dupl_obs_samples[i]]))] = spectra1_dataframe.loc[test_wave]
        dataframe= pd.melt(df)
        dataframe.columns= ['Days','Values']
        sns.boxplot(x="Days", y="Values", data=dataframe, ax=self.mplWidgetSpectral_6.ax)
        self.mplWidgetSpectral_6.canvas.draw()

    def Plot_Feature_Select(self, i):
        self.widget_feature_5.ax.clear()
        # if i < 1:
        #     return
        # label_one = int(self.comboBox_classA_8.currentText().split()[1])
        # label_two = int(self.comboBox_classB_8.currentText().split()[1])
        #
        # avg_spectra1 = self.spectra[:, self.dupl_obs_samples[label_one]:self.dupl_obs_samples[label_one + 1]].mean(
        #     axis=1)
        # std_spectra1 = self.spectra[:, self.dupl_obs_samples[label_one]:self.dupl_obs_samples[label_one + 1]].std(
        #     axis=1)
        #
        # avg_spectra2 = self.spectra[:, self.dupl_obs_samples[label_two]:self.dupl_obs_samples[label_two + 1]].mean(
        #     axis=1)
        # std_spectra2 = self.spectra[:, self.dupl_obs_samples[label_two]:self.dupl_obs_samples[label_two + 1]].std(
        #     axis=1)
        # SDI = np.abs(avg_spectra1 - avg_spectra2) / (std_spectra1 + std_spectra2)
        # self.widget_feature_5.ax.plot(self.wavelength, SDI,
        #                               linewidth=3)
        # self.widget_feature_5.ax.set_ylabel('Spectral Discrimination Index')
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # # specifying horizontal line for SDI<1
        # self.widget_feature_5.ax.axhline(y=1, color='r', linestyle='-', label='SDI<1 --> Poor')
        # # specifying horizontal line for SDI<3
        # self.widget_feature_5.ax.axhline(y=3, color='g', linestyle='-', label='SDI>3 --> Excellent')
        # self.widget_feature_5.ax.legend()
        # self.widget_feature_5.canvas.draw()
        if i < 1:
            return
        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))
        avg_spectra1 = self.df_mean[label_one]
        std_spectra1 = self.df_std[label_one]
        avg_spectra2 = self.df_mean[label_two]
        std_spectra2 = self.df_std[label_two]
        SDI = np.abs(avg_spectra1 - avg_spectra2) / (std_spectra1 + std_spectra2)
        self.widget_feature_5.ax.plot(self.wavelength, SDI,
                                      linewidth=3)
        self.widget_feature_5.ax.set_ylabel('Spectral Discrimination Index')
        self.widget_feature_5.ax.set_xlabel('Wavelength')
        # specifying horizontal line for SDI<1
        self.widget_feature_5.ax.axhline(y=1, color='r', linestyle='-', label='SDI<1 --> Poor')
        # specifying horizontal line for SDI<3
        self.widget_feature_5.ax.axhline(y=3, color='g', linestyle='-', label='SDI>3 --> Excellent')
        # self.widget_feature_5.ax.legend()
        legend = self.mplWidgetSpectral_5.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        legend.set_draggable(True)
        self.widget_feature_5.canvas.draw()

    def OneWayAnova(self, i):
        self.widget_feature_5.ax.clear()
        # if i < 1:
        #     return
        # label_one = int(self.comboBox_classA_8.currentText().split()[1])
        # label_two = int(self.comboBox_classB_8.currentText().split()[1])
        #
        # avg_spectra1 = self.spectra[:, self.dupl_obs_samples[label_one]:self.dupl_obs_samples[label_one + 1]]
        #
        # avg_spectra2 = self.spectra[:, self.dupl_obs_samples[label_two]:self.dupl_obs_samples[label_two + 1]]
        #
        # anova_p = np.zeros((self.wavelength.shape[0]))
        # anova_f = np.zeros((self.wavelength.shape[0]))
        # for i in range(0, self.wavelength.shape[0]):
        #     anova_f[i], anova_p[i] = list(stats.f_oneway(list(avg_spectra1[i, :]), list(avg_spectra2[i, :])))
        # self.widget_feature_5.ax.plot(self.wavelength, anova_p,
        #                               linewidth=3)
        # self.widget_feature_5.ax.set_ylabel('Anova')
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # self.widget_feature_5.canvas.draw()
        if i < 1:
            return
        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))
        spectra1 = self.df[label_one]

        spectra2 = self.df[label_two]

        anova_p = np.zeros((self.wavelength.shape[0]))
        anova_f = np.zeros((self.wavelength.shape[0]))
        for i in range(0, self.wavelength.shape[0]):
            anova_f[i], anova_p[i] = list(stats.f_oneway(list(spectra1[i]), list(spectra2[i])))
        self.widget_feature_5.ax.plot(self.wavelength, anova_p,
                                      linewidth=3)
        self.widget_feature_5.ax.set_ylabel('P-value')
        self.widget_feature_5.ax.set_xlabel('Wavelength')
        self.widget_feature_5.canvas.draw()

    def Tukey_multi_comparison(self):
        self.widget_feature_5.ax.clear()
        self.widget_feature_5.clear()
        test_wave = int(self.x1CoordEdit_10.text())
        df = pd.DataFrame()
        index = self.wavelength
        for i in range(0, self.no_days):
            spectra1 = self.spectra[:, self.dupl_obs_samples[i]:self.dupl_obs_samples[i + 1]]
            spectra1_dataframe = pd.DataFrame(spectra1, index=index)
            df['day' + str(int(self.days_emerg[self.dupl_obs_samples[i]]))] = spectra1_dataframe.loc[test_wave]

        # Stack the data (and rename columns):

        stacked_data = df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                    'level_1': 'treatment',
                                                    0: 'result'})
        # Show the first 8 rows:
        # print (stacked_data.head(8))
        from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                                 MultiComparison)

        # Set up the data for comparison (creates a specialised object)
        MultiComp = MultiComparison(stacked_data['result'],
                                    stacked_data['treatment'])

        # print(MultiComp.tukeyhsd().summary(), "multicomp")

        return MultiComp



    def Kruskal_Wallis(self, i):
        self.widget_feature_5.ax.clear()
        # if i < 1:
        #     return
        # label_one = int(self.comboBox_classA_8.currentText().split()[1])
        # label_two = int(self.comboBox_classB_8.currentText().split()[1])
        #
        # spectra1 = self.spectra[:, self.dupl_obs_samples[label_one]:self.dupl_obs_samples[label_one + 1]]
        #
        # spectra2 = self.spectra[:, self.dupl_obs_samples[label_two]:self.dupl_obs_samples[label_two + 1]]
        #
        # prob = np.zeros((self.wavelength.shape[0]))
        # stat = np.zeros((self.wavelength.shape[0]))
        # for i in range(0, self.wavelength.shape[0]):
        #     stat[i], prob[i] = list(kruskal(list(spectra1[i, :]), list(spectra2[i, :])))
        #
        # # plt.plot(wavelength,anova_f)
        # self.widget_feature_5.ax.plot(self.wavelength, prob)
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # self.widget_feature_5.ax.set_ylabel('prob')
        # # self.widget_feature_5.ax.legend()
        # self.widget_feature_5.canvas.draw()
        if i < 1:
            return
        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))

        spectra1 = self.df[label_one]

        spectra2 = self.df[label_two]

        prob = np.zeros((self.wavelength.shape[0]))
        stat = np.zeros((self.wavelength.shape[0]))
        for i in range(0, self.wavelength.shape[0]):
            stat[i], prob[i] = list(kruskal(list(spectra1[i]), list(spectra2[i])))

        # plt.plot(wavelength,anova_f)
        self.widget_feature_5.ax.plot(self.wavelength, prob)
        self.widget_feature_5.ax.set_xlabel('Wavelength')
        self.widget_feature_5.ax.set_ylabel('prob')
        # self.widget_feature_5.ax.legend()
        self.widget_feature_5.canvas.draw()

    def Mutual_Information(self, i):
        self.widget_feature_5.ax.clear()
        if i < 1:
            return
        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))

        spectra1 = np.array(self.df[label_one].values.tolist())
        spectra2 = np.array(self.df[label_two].values.tolist())
        y1 = np.zeros(spectra1.shape[1])
        y2 = np.ones(spectra2.shape[1])
        X = np.vstack((spectra1.T, spectra2.T))
        y = np.hstack((y1, y2))
        for i in range(0, self.wavelength.shape[0]):
            mutual_info = mutual_info_classif(X, y, discrete_features=False, random_state=None)
        # print(mutual_info, 'mutualinfo')
        # plt.plot(wavelength,anova_f)
        self.widget_feature_5.ax.plot(self.wavelength, mutual_info)
        self.widget_feature_5.ax.set_xlabel('Wavelength')
        self.widget_feature_5.ax.set_ylabel('mutualInfo')
        # self.widget_feature_5.ax.legend()
        self.widget_feature_5.canvas.draw()

    def SDISecondCombo(self, i):
        labels = self.SDIsecond
        # labels = range(0, labels)
        # second_labels = [x for x in labels if x != i - 1]
        second_labels = [x for x in labels if x != labels[i - 1]]
        # second_labels = [x for x in labels]
        self.widget_feature_5.ax.clear()
        self.comboBox_classB_8.clear()
        self.comboBox_classB_8.addItem("--Select--")
        for item in second_labels:
            self.comboBox_classB_8.addItem("Day " + str(item))

    def run(self):
        if (self.lineEdit_2.text() is None) or (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_2.text())):
            self.lineEdit_2.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid", "Kindly enter a valid input file.",
                                           QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
            self.lineEdit.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit.text())):
            self.lineEdit.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid", "Kindly enter a valid file.",
                                           QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_3.text() is None) or (self.lineEdit_3.text() == ""):
            self.lineEdit_3.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Output File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.isdir(os.path.dirname(self.lineEdit_3.text()))):
            self.lineEdit_3.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid", "Kindly enter a valid output path.",
                                           QtWidgets.QMessageBox.Ok)
            return

        try:
            self.inFile = self.lineEdit_2.text()
            self.outFile = self.lineEdit_3.text()
            self.Metafile = self.lineEdit.text()

            self.outputFilename = self.lineEdit_3.text()

            # Read in our image and ROI image
            # img_ds = gdal.Open(inFile, gdal.GA_ReadOnly)
            # roi_ds = gdal.Open(self.Metafile, gdal.GA_ReadOnly)
            df_spectra = pd.read_csv(self.filepath,
                                     header=0, index_col=0)
            self.spectra = df_spectra.to_numpy()
            roi_ds = pd.read_csv(self.metafilepath, header=None, index_col=0)
            # for i in range(1, len(roi_ds)):
            #     roi = roi_ds.iloc[i, 0]
            #     print(roi, 'roi')
            #     self.roi = roi
            df = roi_ds.loc['Days']
            dupl = df.duplicated()
            self.wavelength = df_spectra.index.values

            min = np.min(self.wavelength)
            max = np.max(self.wavelength)


            dupl_obs_samples = dupl[dupl == False].index.values
            self.no_days = np.shape(dupl_obs_samples)[0]
            self.days_emerg = roi_ds.loc['Days'].values
            dupl_obs_samples = np.append(dupl_obs_samples, self.spectra.shape[1] + 1)
            self.dupl_obs_samples = dupl_obs_samples - 1
            self.day = np.unique(self.days_emerg).astype(int)
            self.df_mean = pd.DataFrame()
            self.df_std = pd.DataFrame()
            self.df = pd.DataFrame()
            for i in range(0, self.no_days):
                self.df_mean[str(int(self.days_emerg[dupl_obs_samples[i]]))] = self.spectra[:,
                                                                               self.dupl_obs_samples[i]:
                                                                               self.dupl_obs_samples[
                                                                                   i + 1]].mean(axis=1)
                self.df_std[str(int(self.days_emerg[dupl_obs_samples[i]]))] = self.spectra[:,
                                                                              self.dupl_obs_samples[i]:
                                                                              self.dupl_obs_samples[
                                                                                  i + 1]].std(axis=1)
                self.df[str(int(self.days_emerg[dupl_obs_samples[i]]))] = self.spectra[:,
                                                                          self.dupl_obs_samples[i]:
                                                                          self.dupl_obs_samples[
                                                                              i + 1]].tolist()
            if (self.lineEdit.text() == ""):
                messageDisplay = "metadataentry!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.lineEdit_2.text() == ""):
                messageDisplay = "Cannot leave input empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.lineEdit_3.text() == ""):
                messageDisplay = "Cannot leave output empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.x1CoordEdit_10.text() == ""):
                messageDisplay = "Cannot leave Wavelength empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.x1CoordEdit_11.text() == ""):
                messageDisplay = "Cannot leave Baseline empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            print("In: " + self.inFile)
            print("Out: " + self.outFile)
            print("Meta data: " + self.Metafile)
            print("Running...")
            import seaborn as sns

            if self.rb_sep_all_5.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.plotCV()
                QApplication.restoreOverrideCursor()
            # if self.rb_sep_cv_5.isChecked():
            #     QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            #     self.plotCV()
            #     QApplication.restoreOverrideCursor()
            if self.rb_sep_compare_5.isChecked():
                # print('Current Index :',self.comboBox_26.currentIndex())
                self.twoClassPlot(self.comboBox_26.currentIndex())
                # self.toggleComboboxes()
                # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.comboBox_25.clear()
                # self.comboBox_25.addItem("--Select--")
                # # for i in np.unique(self.day):
                # #     self.comboBox_25.addItem("Day " + str(i))
                # # self.roi_ds = np.unique(self.day)
                #
                # # for i in range(0, self.no_days):
                # #     self.comboBox_25.addItem("class " + str(i))
                # # self.roi_ds = range(0, self.no_days)
                # for i in self.df.columns:
                #     self.comboBox_25.addItem("Day " + i)
                # self.roi_ds = self.df.columns
                #
                # QApplication.restoreOverrideCursor()
                # self.comboBox_25.currentIndexChanged.connect(self.updateSecondCombo)
                # self.comboBox_26.currentIndexChanged.connect(self.twoClassPlot)
            if self.radioClassDistribute.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.comboBox_3.clear()
                # self.comboBox_3.addItem("--Select--")
                # for i in self.df.columns:
                #     self.comboBox_3.addItem("Day " + i)
                # self.comboBox_3.currentIndexChanged.connect(self.Plot_Continum_removal)
                # QApplication.restoreOverrideCursor()

                self.Plot_Continum_removal()
                QApplication.restoreOverrideCursor()

            if self.radioFTEST_5.isChecked():
                # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.comboBox_classA_8.clear()
                # self.comboBox_classA_8.addItem("--Select--")
                # # for i in range(0, self.no_days):
                # #     self.comboBox_classA_8.addItem("class " + str(i))
                # # self.SDIsecond = range(0, self.no_days)
                # for i in self.df.columns:
                #     self.comboBox_classA_8.addItem("Day " + i)
                # self.SDIsecond = self.df.columns
                # QApplication.restoreOverrideCursor()
                # self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                # self.comboBox_classB_8.currentIndexChanged.connect(self.Plot_Feature_Select)

                self.Plot_Feature_Select(self.comboBox_classB_8.currentIndex())

            if self.radioSEP_5.isChecked():
                # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.comboBox_classA_8.clear()
                # self.comboBox_classA_8.addItem("--Select")
                # # for i in range(0, self.no_days):
                # #     self.comboBox_classA_8.addItem("class " + str(i))
                # # self.SDIsecond = range(0, self.no_days)
                # for i in self.df.columns:
                #     self.comboBox_classA_8.addItem("Day " + i)
                # self.SDIsecond = self.df.columns
                # self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                # self.comboBox_classB_8.currentIndexChanged.connect(self.OneWayAnova)

                self.OneWayAnova(self.comboBox_classB_8.currentIndex())

            if self.radioMUTUAL_5.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                QApplication.restoreOverrideCursor()

                in_value = int(self.x1CoordEdit_10.text())

                if in_value < min or in_value > max:
                    messageDisplay = "Invalid Wavelength Value. Value should be between " + str(min) + " and " + str(
                        max)
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                MultiComp=self.Tukey_multi_comparison()
                dlg = Dialog()
                dlg.textBrowser.append(MultiComp.tukeyhsd().summary().as_text())
                dlg.exec_()

            if self.radioTREE_5.isChecked():
                # QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.comboBox_classA_8.clear()
                # self.comboBox_classA_8.addItem("--Select")
                # # for i in range(0, self.no_days):
                # #     self.comboBox_classA_8.addItem("class " + str(i))
                # # self.SDIsecond = range(0, self.no_days)
                # for i in self.df.columns:
                #     self.comboBox_classA_8.addItem("Day " + i)
                # self.SDIsecond = self.df.columns
                # self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                # self.comboBox_classB_8.currentIndexChanged.connect(self.Kruskal_Wallis)
                self.Kruskal_Wallis(self.comboBox_classB_8.currentIndex())


            if self.radioFeatureVar.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # self.comboBox_3.clear()
                # self.comboBox_3.addItem("--Select--")
                # for i in self.df.columns:
                #     self.comboBox_3.addItem("Day " + i)
                # self.comboBox_3.currentIndexChanged.connect(self.Band_plot)
                QApplication.restoreOverrideCursor()

                in_value = int(self.x1CoordEdit_13.text())

                if in_value < min or in_value > max:
                    messageDisplay = "Invalid Wavelength Value. Value should be between " + str(min) + " and " + str(
                        max)
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                self.Band_plot()

            # if self.radioButton_bandnorm_5.isChecked():
            #     QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            #     self.comboBox_classA_1.clear()
            #     self.comboBox_classA_1.addItem("Class 1")
            #     for i in (np.unique(self.roi[self.roi > 1])):
            #         self.comboBox_classA_1.addItem("Class " + str(i))
            #     self.comboBox_classB_1.clear()
            #     self.comboBox_classB_1.addItem("OtherClasses")
            #     for i in (np.unique(self.roi[self.roi > 0])):
            #         self.comboBox_classB_1.addItem("Class " + str(i))
            #     QApplication.restoreOverrideCursor()
            #     self.comboBox_classB_1.currentIndexChanged.connect(self.Plot_feature_sep_mat)
            # if self.radioButton_bandwise_5.isChecked():
            #     QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            #     self.comboBox_classA_1.clear()
            #     self.comboBox_classA_1.addItem("Class 1")
            #     for i in (np.unique(self.roi[self.roi > 1])):
            #         self.comboBox_classA_1.addItem("Class " + str(i))
            #     self.comboBox_classB_1.clear()
            #     self.comboBox_classB_1.addItem("OtherClasses")
            #     for i in (np.unique(self.roi[self.roi > 0])):
            #         self.comboBox_classB_1.addItem("Class " + str(i))
            #     QApplication.restoreOverrideCursor()
            #     self.comboBox_classB_1.currentIndexChanged.connect(self.Plot_feature_sep_mat)

            print("Completed!!!")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            print(e)


def feature_select(X, y, algo):
    from sklearn.feature_selection import f_classif, mutual_info_regression
    from sklearn.feature_selection import SelectFromModel

    if algo == "classif":
        f_test, pval = f_classif(X, y)
        f_test /= np.max(f_test)
        score = -np.log10(pval)
        score /= score.max()
        return f_test, score

    if algo == "mutual":
        mi = mutual_info_regression(X, y)
        mi /= np.max(mi)
        pval = 0 * mi
        return mi, pval

    # if algo=="L1":
    #     from sklearn.svm import LinearSVC
    #     lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    #     model = SelectFromModel(lsvc, prefit=True)
    #     X_new = model.transform(X)
    #     pval=0*lsvc.feature_importances_
    #     return lsvc.feature_importances_, pval

    if algo == "Tree":
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        pval = 0 * clf.feature_importances_
        return clf.feature_importances_, pval

    if algo == "SEP":
        m1 = X[np.ndarray.flatten(y == 0), :].mean(axis=0)
        m2 = X[np.ndarray.flatten(y == 1), :].mean(axis=0)
        s1 = X[np.ndarray.flatten(y == 0), :].std(axis=0)
        s2 = X[np.ndarray.flatten(y == 1), :].std(axis=0)
        SDI = (np.abs(m1 - m2)) / (s1 + s2)
        SDI /= np.max(SDI)
        return SDI, SDI * 0


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    # QSizePolicy sretain=Form.sizePolicy()
    # sretain.setRetainSizeWhenHidden(True)
    # sretain.setSizePolicy()
    ui = Timeseries()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
