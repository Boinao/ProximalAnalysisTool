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
from PyQt5.QtWidgets import QFileDialog, QApplication, QDialog
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QCursor

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from Ui.VisualizerUI import Ui_Form
import os
import math
import scipy.stats as stats
from math import exp, sqrt, log
from PIL import Image
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
import importlib
from sklearn.feature_selection import mutual_info_classif
import pysptools.spectro as spectro
from modules.PandasModel import PandasModel
# Import the Py6S module
from Py6S import SixS

import matplotlib.pyplot as plt

# from . import GdalTools_utils as Utils
POSTFIX = '_Visualizer'
from os import path
import seaborn as sns
import matplotlib.patches as mpatches
from modules import Utils


from PyQt5 import uic
pluginPath = os.path.split(os.path.dirname(__file__))[0]
# print(pluginPath)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'Ui', 'dialogResultViewer.ui'))


'''
Control variable for visualizing Time series or normal data
'''
NORMAL=1
TIME_SERIES=2

# from Ui.dialogResultViewer import Ui_Dialog

class Dialog(BASE, WIDGET):
    def __init__(self, title='Result Viewer', parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.setupUi(self)


class Visualizer(Ui_Form):

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
        super(Visualizer, self).setupUi(Form)
        self.Form = Form

        self.baselineLe.setText("0.0")
        self.baselineLe.setVisible(False)
        self.label_73.setVisible(False)

        self.connectWidgets()
        self.metafilepath = None
        self.compareCmbLoaded=False
        self.compareBandSepLoaded=False
        self.compareStatistics=False
        self.statsSingle=False

    def connectWidgets(self):
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_5.clicked.connect(lambda: self.MetadatabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())
        self.plot2ClassesRb.toggled.connect(self.toggleRadio_rb_sep_compare)
        # self.radioClassDistribute.toggled.connect(self.toggleRadio_ClassDistribute)
        self.sdiRb.toggled.connect(self.toggleSDI)
        self.annovaRb.toggled.connect(self.toggleAnnova)
        self.tukeyRb.toggled.connect(self.toggleTukey)
        self.kruskalRb.toggled.connect(self.toggleKruskal)
        self.plotDistnRb.toggled.connect(self.toggleDistn)
        self.plotStatsRb.toggled.connect(self.toggleRadioStatistics)
        self.plotOneParamRb.toggled.connect(self.toggleRadioSingle)

        self.x1CoordEdit_10.setValidator(QIntValidator())
        self.wavelengthLe.setValidator(QIntValidator())

    def clearButtonGrp(self):
        self.buttonGroup.setExclusive(False)
        self.buttonGroup.checkedButton().setChecked(False)
        self.buttonGroup.setExclusive(True)


    def toggleRadio_rb_sep_compare(self):
        self.comboBox_25.setEnabled(self.plot2ClassesRb.isChecked())
        self.comboBox_26.setEnabled(self.plot2ClassesRb.isChecked())

        # Read in our image and ROI image
        # img_ds = gdal.Open(inFile, gdal.GA_ReadOnly)
        # roi_ds = gdal.Open(self.Metafile, gdal.GA_ReadOnly)
        if self.plot2ClassesRb.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not self.compareCmbLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                classes=metadata.loc['class_label'].unique().astype(int)


                self.comboBox_25.clear()
                self.comboBox_26.clear()
                self.comboBox_25.addItem("--Select--")
                for i in classes:
                    self.comboBox_25.addItem("Class " + str(i))

                QApplication.restoreOverrideCursor()
                self.roi_ds=classes
                self.comboBox_25.currentIndexChanged.connect(self.updateSecondCombo)
                self.compareCmbLoaded=True
            # self.comboBox_26.currentIndexChanged.connect(self.twoClassPlot)





    # def toggleRadio_ClassDistribute(self):
    #     self.x1CoordEdit_11.setEnabled(self.radioClassDistribute.isChecked())

    def toggleRadio_FRA(self):
        self.comboBox_28.setEnabled(self.radioFRA_5.isChecked())

    def toggleRadio_SAM(self):
        self.comboBox_28.setEnabled(self.radioSAM_5.isChecked())


    def loadDependentCmb(self, sourceRb,flag_name,kind):

        self.comboBox_classA_8.setEnabled(sourceRb.isChecked())
        self.comboBox_classB_8.setEnabled(sourceRb.isChecked())

        if self.plot2ClassesRb.isChecked() or self.sdiRb.isChecked() or self.annovaRb.isChecked() or self.kruskalRb.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty',
                                                  QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not flag_name:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                self.comboBox_classA_8.clear()
                self.comboBox_classB_8.clear()
                self.comboBox_classA_8.addItem("--Select--")
                if kind == NORMAL:
                    classes = metadata.loc['class_label'].unique().astype(int)
                    for i in classes:
                        self.comboBox_classA_8.addItem("Class " + str(i))
                elif kind== TIME_SERIES:
                    classes = metadata.loc['Days'].unique().astype(int)
                    for i in classes:
                        self.comboBox_classA_8.addItem("Day " + str(i))
                QApplication.restoreOverrideCursor()
                self.SDIsecond = classes
                self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                flag_name = True


    def toggleSDI(self):
        self.loadDependentCmb(self.sdiRb, self.compareBandSepLoaded, NORMAL)
        # self.comboBox_classA_8.setEnabled(self.sdiRb.isChecked())
        # self.comboBox_classB_8.setEnabled(self.sdiRb.isChecked())
        #
        #
        # if self.sdiRb.isChecked() or self.annovaRb.isChecked() or self.kruskalRb.isChecked():
        #     if self.metafilepath is None:
        #         # print('Input data or metadatafile may be empty')
        #         QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
        #         self.clearButtonGrp()
        #         return
        #
        #     if not self.compareBandSepLoaded:
        #         QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        #         metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
        #         classes=metadata.loc['class_label'].unique().astype(int)
        #
        #
        #         self.comboBox_classA_8.clear()
        #         self.comboBox_classB_8.clear()
        #         self.comboBox_classA_8.addItem("--Select--")
        #         for i in classes:
        #             self.comboBox_classA_8.addItem("Class " + str(i))
        #
        #         QApplication.restoreOverrideCursor()
        #         self.SDIsecond=classes
        #         self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
        #         self.compareBandSepLoaded=True
            # self.comboBox_26.currentIndexChanged.connect(self.twoClassPlot)

    def toggleAnnova(self):
        self.comboBox_classA_8.setEnabled(self.annovaRb.isChecked())
        self.comboBox_classB_8.setEnabled(self.annovaRb.isChecked())

        if self.sdiRb.isChecked() or self.annovaRb.isChecked() or self.kruskalRb.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not self.compareBandSepLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                classes=metadata.loc['class_label'].unique().astype(int)


                self.comboBox_classA_8.clear()
                self.comboBox_classB_8.clear()
                self.comboBox_classA_8.addItem("--Select--")
                for i in classes:
                    self.comboBox_classA_8.addItem("Class " + str(i))

                QApplication.restoreOverrideCursor()
                self.SDIsecond=classes
                self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                self.compareBandSepLoaded=True

    def toggleTukey(self):
        self.x1CoordEdit_10.setEnabled(self.tukeyRb.isChecked())





    def toggleKruskal(self):
        self.comboBox_classA_8.setEnabled(self.kruskalRb.isChecked())
        self.comboBox_classB_8.setEnabled(self.kruskalRb.isChecked())

        if self.sdiRb.isChecked() or self.annovaRb.isChecked() or self.kruskalRb.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty', QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            if not self.compareBandSepLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
                classes=metadata.loc['class_label'].unique().astype(int)


                self.comboBox_classA_8.clear()
                self.comboBox_classB_8.clear()
                self.comboBox_classA_8.addItem("--Select--")
                for i in classes:
                    self.comboBox_classA_8.addItem("Class " + str(i))

                QApplication.restoreOverrideCursor()
                self.SDIsecond=classes
                self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                self.compareBandSepLoaded=True



    def toggleRadio_pairplot(self):
        self.comboBox_29.setEnabled(self.radioPairplot_5.isChecked())
        self.comboBox_30.setEnabled(self.radioPairplot_5.isChecked())

    # def toggleRadio_featureselect(self):
    #     self.comboBox_classA.setEnabled(self.radioButton_feature.isChecked())
    #     self.comboBox_classB.setEnabled(self.radioButton_feature.isChecked())

    def toggleDistn(self):
        self.wavelengthLe.setEnabled(self.plotDistnRb.isChecked())
        # self.comboBox_3.setEnabled(self.plotDistnRb.isChecked())

    def toggleRadio_TSNE(self):
        self.x1CoordEdit_9.setEnabled(self.radioButton_TSNE_5.isChecked())

    def toggleRadio_bandnorm(self):
        # self.x1CoordEdit_10.setEnabled(self.radioButton_bandnorm_5.isChecked())
        self.comboBox_classA_9.setEnabled(self.radioButton_bandnorm_5.isChecked())
        self.comboBox_classB_9.setEnabled(self.radioButton_bandnorm_5.isChecked())

    def toggleRadio_bandwise(self):
        # self.x1CoordEdit_2.setEnabled(self.radioButton_bandwise.isChecked())
        self.comboBox_classA_9.setEnabled(self.radioButton_bandwise_5.isChecked())
        self.comboBox_classB_9.setEnabled(self.radioButton_bandwise_5.isChecked())

    # def onLayersChanged(self):
    #     self.inSelector.setLayers(Utils.LayerRegistry.instance().getRasterLayers())
    def toggleRadioStatistics(self):
        self.comboBox_31.setEnabled((self.plotStatsRb.isChecked()))
        if self.plotStatsRb.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty',
                                                  QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            self.meta_transpose=None
            metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)

            if not self.compareStatistics:
                self.comboBox_31.clear()
                self.comboBox_31.addItem("--Select")
                meta_copy = metadata.T
                self.meta_transpose = metadata.T
                # self.meta_transpose.drop(columns=['class'], axis=1, inplace=True)
                meta_copy.drop(["Spectra ID", "class_label", "class"], axis=1, inplace=True)
                for i in meta_copy.columns:
                    self.comboBox_31.addItem(str(i))

                self.compareStatistics=True

            # self.comboBox_31.currentIndexChanged.connect(self.Plot_Statistics)


    def toggleRadioSingle(self):
        self.comboBox_33.setEnabled((self.plotOneParamRb.isChecked()))
        if self.plotOneParamRb.isChecked():
            if self.metafilepath is None:
                # print('Input data or metadatafile may be empty')
                QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty',
                                                  QtWidgets.QMessageBox.Ok)
                self.clearButtonGrp()
                return

            metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
            if not self.statsSingle:
                self.comboBox_33.clear()
                self.comboBox_33.addItem("--Select")
                meta_copy = metadata.T
                self.meta_transpose = metadata.T
                # self.meta_transpose.drop(columns=['class'], axis=1, inplace=True)
                meta_copy.drop(["Spectra ID", "class_label", "class"], axis=1, inplace=True)
                for i in meta_copy.columns:
                    self.comboBox_33.addItem(str(i))

                self.statsSingle=True

    def toggleComboboxes(self):
        self.comboBox_25.setEnabled(self.plot2ClassesRb.isChecked())
        self.comboBox_26.setEnabled(self.plot2ClassesRb.isChecked())

    def stateChanged1(self):
        self.lineEdit.setEnabled(True)

    def stateChanged2(self):
        self.lineEdit.setEnabled(True)

    def flushFlag(self):

        self.compareCmbLoaded = False
        self.compareBandSepLoaded = False
        self.compareStatistics = False
        self.statsSingle=False

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

            self.outputFilename = (os.path.dirname(fname)) + "/Output" + POSTFIX + ".csv"
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

            # self.outputFilename = (str(self.curdir)) + "/Output" + POSTFIX + ".csv"
            # self.lineEdit_3.setText(self.outputFilename)
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

    def saveas(self):
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', self.curdir, '*.csv')
        if self.outputFilename:
            self.lineEdit_3.setText(self.outputFilename)

        return self.outputFilename


    def plotAllMeanStd(self):
        self.mplWidgetSpectral_5.clear()
        df_spectra = pd.read_csv(self.filepath, header=0, index_col=0)
        df_metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)

        # print(df_spectra)

        no_sample_avg = 10
        spectra = df_spectra.to_numpy()
        wavelength = df_spectra.index.values
        crop_param = df_metadata.index
        spectra_id = df_metadata.loc['Spectra ID'].values
        samples = df_metadata.loc['Spectra ID'].values
        # metadata = df_metadata.to_numpy()
        days_emerg = df_metadata.loc['class_label'].values
        # print(df_metadata, "metadata")

        # //Compute Mean Spectra Based on the No. of Replications
        df = df_metadata.loc['class_label']
        dupl = df.duplicated()
        dupl_obs_samples = dupl[dupl == False].index.values

        no_days = np.shape(dupl_obs_samples)[0]
        avg_spectra = np.zeros((wavelength.shape[0], no_days))

        dupl_obs_samples = np.append(dupl_obs_samples, spectra.shape[1])
        dupl_obs_samples = dupl_obs_samples - 1
        df = df_metadata.loc['class_label']
        dupl = df.duplicated()
        dupl_obs_samples = dupl[dupl == False].index.values


        # //Visulaize the Spectra using Fill Curve
        df = df_metadata.loc['class_label']
        dupl = df.duplicated()
        dupl_obs_samples = dupl[dupl == False].index.values

        no_days = np.shape(dupl_obs_samples)[0]
        avg_spectra = np.zeros((wavelength.shape[0], no_days))

        days_emerg = df_metadata.loc['class_label'].values
        self.crop = df_metadata.loc['class'].values
        # print(days_emerg, 'days')
        # print(self.crop, 'crop')
        dupl_obs_samples = np.append(dupl_obs_samples, spectra.shape[1] + 1)
        dupl_obs_samples = dupl_obs_samples - 1

        avg_spectra_ = np.zeros((wavelength.shape[0], no_days))
        std_spectra_ = np.zeros((wavelength.shape[0], no_days))

        for i in range(0, no_days):
            avg_spectra = spectra[:, dupl_obs_samples[i]:dupl_obs_samples[i + 1]].mean(axis=1)
            std_spectra = spectra[:, dupl_obs_samples[i]:dupl_obs_samples[i + 1]].std(axis=1)
            avg_spectra_[:, i] = avg_spectra
            std_spectra_[:, i] = std_spectra
            # print(dupl_obs_samples, no_days, avg_spectra[0])

            lower = avg_spectra - std_spectra
            upper = avg_spectra + std_spectra
            self.mplWidgetSpectral_5.ax.plot(wavelength, avg_spectra,
                                             label='Class = ' + str(self.crop[dupl_obs_samples[i]]), linewidth=3)
            legend=self.mplWidgetSpectral_5.ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
            legend.set_draggable(True)
            self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
            self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
            self.mplWidgetSpectral_5.ax.fill_between(self.wavelength, lower, upper, alpha=0.5)
        self.mplWidgetSpectral_5.canvas.draw()


        # print(avg_spectra_.shape,std_spectra_.shape)
        result = np.hstack((avg_spectra_, std_spectra_))
        # print(result.shape)
        df1 = pd.DataFrame(result, index=wavelength)
        # print([str(int(days_emerg[dupl_obs_samples[i]])) for i in range(no_days)]*2)
        head = ['Class_Mean_' + str(self.crop[dupl_obs_samples[i]]) for i in range(no_days)] + ['Class_Std_' + str(self.crop[dupl_obs_samples[i]]) for i in range(no_days)]
        df1.to_csv(self.outputFilename, header=head, index=True)

    def plot2Classes(self):
        self.mplWidgetSpectral_5.clear()

        if self.comboBox_25.currentIndex()<1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class A', QtWidgets.QMessageBox.Ok)
            self.comboBox_25.setFocus()
            return

        if self.comboBox_26.currentIndex()<1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class B', QtWidgets.QMessageBox.Ok)
            self.comboBox_26.setFocus()
            return


        label_one = str(int(self.comboBox_25.currentText().split()[1]))
        label_two = str(int(self.comboBox_26.currentText().split()[1]))
        spectra1 = self.df_mean[label_one]
        spectra2 = self.df_mean[label_two]
        self.mplWidgetSpectral_5.ax.plot(self.wavelength, spectra1, label='Class = ' + str(self.crop[self.dupl_obs_samples[int(label_one)-1]]),
                                         # label='Class = ' + str(label_one),
                                         linewidth=3)
        self.mplWidgetSpectral_5.ax.plot(self.wavelength, spectra2, label='Class = ' + str(self.crop[self.dupl_obs_samples[int(label_two)-1]]),
                                         # label='Class = ' + str(label_two),
                                         linewidth=3)
        self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
        self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
        # self.mplWidgetSpectral_5.ax.legend()
        legend = self.mplWidgetSpectral_5.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        legend.set_draggable(True)
        self.mplWidgetSpectral_5.canvas.draw()

    def plotContinuum(self):
        self.mplWidgetSpectral_5.clear()
        baseline = float(self.x1CoordEdit_11.text())
        for i in range(0, self.no_days):
            spectra1 = self.spectra[:, self.dupl_obs_samples[i]:self.dupl_obs_samples[i + 1]].mean(axis=1).tolist()
            wvl = self.wavelength.tolist()

            fea = spectro.FeaturesConvexHullQuotient(spectra1, wvl, baseline=baseline)
            self.mplWidgetSpectral_5.ax.plot(self.wavelength, fea.crs,
                                             label='Class = ' + str(
                                                 self.crop[self.dupl_obs_samples[i]]))
                                             # label='Class = ' + str(int(self.days_emerg[self.dupl_obs_samples[i]])))
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
            self.comboBox_26.addItem("Class " + str(item))

    def SDISecondCombo(self, i):
        labels = self.SDIsecond
        # labels = range(0, labels)
        # second_labels = [x for x in labels if x != i - 1]
        second_labels = [x for x in labels if x != labels[i - 1]]
        # second_labels = [x for x in labels]
        self.mplWidgetSep.ax.clear()
        self.comboBox_classB_8.clear()
        self.comboBox_classB_8.addItem("--Select--")
        for item in second_labels:
            self.comboBox_classB_8.addItem("Class " + str(item))

    def Band_plot(self):

        self.mplWidgetDist.ax.clear()
        test_wave = int(self.wavelengthLe.text())
        # if i < 1:
        #     return
        # label_one = str(int(self.comboBox_3.currentText().split()[1]))
        df = pd.DataFrame()
        index = self.wavelength

        for i in range(0, self.no_days):
            spectra1 = self.spectra[:, self.dupl_obs_samples[i]:self.dupl_obs_samples[i + 1]]
            spectra1_dataframe = pd.DataFrame(spectra1, index=index)
            # print(spectra1_dataframe)
            df[' ' + str(self.crop[self.dupl_obs_samples[i]])] = spectra1_dataframe.loc[test_wave]
        dataframe = pd.melt(df)
        # print(dataframe)
        dataframe.columns = ['Crops', 'Values']
        sns.boxplot(x="Crops", y="Values", data=dataframe, ax=self.mplWidgetDist.ax)
        self.mplWidgetDist.canvas.draw()

    def plotSDI(self):
        self.mplWidgetSep.clear()

        if self.comboBox_classA_8.currentIndex()<1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class A', QtWidgets.QMessageBox.Ok)
            self.comboBox_classA_8.setFocus()
            return

        if self.comboBox_classB_8.currentIndex()<1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class B', QtWidgets.QMessageBox.Ok)
            self.comboBox_classB_8.setFocus()
            return
        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))
        avg_spectra1 = self.df_mean[label_one]
        std_spectra1 = self.df_std[label_one]
        avg_spectra2 = self.df_mean[label_two]
        std_spectra2 = self.df_std[label_two]
        SDI = np.abs(avg_spectra1 - avg_spectra2) / (std_spectra1 + std_spectra2)
        self.mplWidgetSep.ax.plot(self.wavelength, SDI,
                                      linewidth=3)
        self.mplWidgetSep.ax.set_ylabel('Spectral Discrimination Index')
        self.mplWidgetSep.ax.set_xlabel('Wavelength')
        # specifying horizontal line for SDI<1
        self.mplWidgetSep.ax.axhline(y=1, color='r', linestyle='-', label='SDI<1 --> Poor')
        # specifying horizontal line for SDI<3
        self.mplWidgetSep.ax.axhline(y=3, color='g', linestyle='-', label='SDI>3 --> Excellent')
        # self.mplWidgetSep.ax.legend()
        legend = self.mplWidgetSep.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        legend.set_draggable(True)
        self.mplWidgetSep.canvas.draw()

    def OneWayAnova(self):
        self.mplWidgetSep.clear()

        if self.comboBox_classA_8.currentIndex() < 1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class A',
                                              QtWidgets.QMessageBox.Ok)
            self.comboBox_classA_8.setFocus()
            return

        if self.comboBox_classB_8.currentIndex() < 1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class B',
                                              QtWidgets.QMessageBox.Ok)
            self.comboBox_classB_8.setFocus()
            return
        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))
        spectra1 = self.df[label_one]

        spectra2 = self.df[label_two]

        anova_p = np.zeros((self.wavelength.shape[0]))
        anova_f = np.zeros((self.wavelength.shape[0]))
        for i in range(0, self.wavelength.shape[0]):
            anova_f[i], anova_p[i] = list(stats.f_oneway(list(spectra1[i]), list(spectra2[i])))
        self.mplWidgetSep.ax.plot(self.wavelength, anova_p,
                                      linewidth=3)
        self.mplWidgetSep.ax.set_ylabel('P-value')
        self.mplWidgetSep.ax.set_xlabel('Wavelength')
        self.mplWidgetSep.canvas.draw()

    def Tukey_multi_comparison(self):
        self.mplWidgetSep.clear()
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
        return MultiComp

        # print(MultiComp.tukeyhsd().summary(), "multicomp")

    def Kruskal_Wallis(self):
        self.mplWidgetSep.clear()

        if self.comboBox_classA_8.currentIndex() < 1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class A',
                                              QtWidgets.QMessageBox.Ok)
            self.comboBox_classA_8.setFocus()
            return

        if self.comboBox_classB_8.currentIndex() < 1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first Class B',
                                              QtWidgets.QMessageBox.Ok)
            self.comboBox_classB_8.setFocus()

        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))

        spectra1 = self.df[label_one]

        spectra2 = self.df[label_two]

        prob = np.zeros((self.wavelength.shape[0]))
        stat = np.zeros((self.wavelength.shape[0]))
        for i in range(0, self.wavelength.shape[0]):
            stat[i], prob[i] = list(kruskal(list(spectra1[i]), list(spectra2[i])))

        # plt.plot(wavelength,anova_f)
        self.mplWidgetSep.ax.plot(self.wavelength, prob)
        self.mplWidgetSep.ax.set_xlabel('Wavelength')
        self.mplWidgetSep.ax.set_ylabel('prob')
        # self.widget_feature_5.ax.legend()
        self.mplWidgetSep.canvas.draw()

    def Mutual_Information(self, i=None):
        self.mplWidgetSep.clear()
        if i < 1:
            return
        label_one = str(int(self.comboBox_classA_8.currentText().split()[1]))
        label_two = str(int(self.comboBox_classB_8.currentText().split()[1]))

        # print('here1')
        spectra1 = np.array(self.df[label_one].values.tolist())
        spectra2 = np.array(self.df[label_two].values.tolist())
        y1 = np.zeros(spectra1.shape[1])
        y2 = np.ones(spectra2.shape[1])
        X = np.vstack((spectra1.T, spectra2.T))
        y = np.hstack((y1, y2))
        # print('here2')
        for i in range(0, self.wavelength.shape[0]):
            mutual_info = mutual_info_classif(X, y, discrete_features=False, random_state=None)
        print(mutual_info, 'mutualinfo')
        # plt.plot(wavelength,anova_f)
        self.mplWidgetSep.ax.plot(self.wavelength, mutual_info)
        self.mplWidgetSep.ax.set_xlabel('Wavelength')
        self.mplWidgetSep.ax.set_ylabel('mutualInfo')
        # self.widget_feature_5.ax.legend()
        self.mplWidgetSep.canvas.draw()

    def Plot_Class_Correlate(self):
        self.widget_corr_5.clear()
        self.widget_corr_5.ax.clear()
        df_spectra = pd.read_csv(self.filepath,
                                 header=0, index_col=0)

        indxs = df_spectra.T.shape
        # # Compute the correlation matrix
        corr = df_spectra.T.corr()
        # print(corr)
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        # Draw the heatmap with the mask and correct aspect ratio
        g=sns.heatmap(corr, cmap='RdYlGn', vmax=1,
                    xticklabels=max(5, indxs[1] // 10),
                    yticklabels=max(5, indxs[1] // 10),
                    square=False, cbar=False, mask=mask, ax=self.widget_corr_5.ax)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        self.widget_corr_5.ax.set_title("Wavelength Correlation")
        self.widget_corr_5.ax.set_xlabel("Wavelength")
        self.widget_corr_5.ax.set_ylabel("Wavelength")
        # plt.colorbar()  will do
        self.widget_corr_5.canvas.draw()

    def Plot_Statistics(self, i=None):
        self.widget_corr_6.ax.clear()
        try:

            # df_spectra = pd.read_csv(self.filepath,
            #                          header=0, index_col=0)
            value=pd.DataFrame()
            # if i < 1:
            #     return
            if self.comboBox_31.currentIndex() < 1:
                QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select property',
                                                  QtWidgets.QMessageBox.Ok)
                self.comboBox_31.setFocus()
                return

            label_one = str(self.comboBox_31.currentText())

            data = self.meta_transpose.drop(["Spectra ID", "class_label", "class"], axis=1)  # self.meta_transpose
            params = data.iloc[:,:].astype('float')
            params['class'] = self.meta_transpose[['class']]
            # df1 = self.meta_transpose.iloc[:, 3::]
            # data = self.meta_transpose.drop(columns=['class'], axis=1, inplace=True) #self.meta_transpose
            # params = data.iloc[:, 3:].astype('float')
            # params['class'] = data[['class']]
            sns.boxplot(x='class', y=label_one, data=params, ax=self.widget_corr_6.ax)
            self.widget_corr_6.canvas.draw()
            print(params[label_one].describe())
            params[label_one].describe().to_csv(self.outputFilename)

        except Exception as e:
            import traceback
            print(e, traceback.format_exc())

    def plotSingleparameter(self,i=None):
        self.widget_corr_7.ax.clear()
        try:

            # df_spectra = pd.read_csv(self.filepath,
            #                          header=0, index_col=0)
            value = pd.DataFrame()
            # if i < 1:
            #     return
            if self.comboBox_33.currentIndex() < 1:
                QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select property',
                                                  QtWidgets.QMessageBox.Ok)
                self.comboBox_33.setFocus()
                return
            label_one = str(self.comboBox_33.currentText())

            data = self.meta_transpose.drop(["Spectra ID", "class_label", "class"], axis=1)  # self.meta_transpose
            params = data.iloc[:, :].astype('float')
            params['class'] = self.meta_transpose[['class']]

            # data = self.meta_transpose
            # params = data.iloc[:, 3:].astype('float')
            # params['class'] = data[['class']]

            from scipy.stats import norm
            x=params[[label_one]].to_numpy()
            sns.distplot(x, fit=norm, kde=False,ax=self.widget_corr_7.ax)
            # df1 = self.meta_transpose.iloc[:, 3::]
            # df = pd.DataFrame()
            # self.widget_corr_7.ax.plot(parameter)
            self.widget_corr_7.canvas.draw()
        except Exception as e:
            import traceback
            print(e, traceback.format_exc())

    def plotMultipleparameter(self):
        try:
            self.widget_corr_7.ax.clear()
            data = self.meta_transpose
            # df1 = self.meta_transpose.iloc[:, 3::]


            data = self.meta_transpose.drop(["Spectra ID", "class_label", "class"], axis=1)  # self.meta_transpose
            params = data.iloc[:, :].astype('float')
            params['class'] = self.meta_transpose[['class']]

            sns.boxplot(data=data.iloc[:, :].astype('float'), ax=self.widget_corr_7.ax)
            self.widget_corr_7.canvas.draw()


            # params = data.iloc[:, 3:].astype('float')
            # params['class'] = data[['class']]
            sns.pairplot(params, hue="class")
            print(params.describe())
            plt.show()


        except Exception as e:
            import traceback
            print(e, traceback.format_exc())



    def run(self):

        if (self.lineEdit_2.text() == ""):
            messageDisplay = "Cannot leave input empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_2.text())):
            self.lineEdit_2.setFocus()
            messageDisplay = "Path does not exist : " + self.lineEdit_2.text()
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return



        if (self.lineEdit.text() == ""):
            messageDisplay = "please upload metadata file!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit.text())):
            self.lineEdit.setFocus()
            messageDisplay = "Path does not exist : " + self.lineEdit.text()
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_3.text() == ""):
            messageDisplay = "Cannot leave output empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.isdir(os.path.dirname(self.lineEdit_3.text()))):
            self.lineEdit_3.setFocus()
            messageDisplay = "Kindly enter a valid output path."
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        # if self.tabWidget_4.currentIndex()==0:
        #     if self.radioClassDistribute.isChecked():
        #         if (self.x1CoordEdit_11.text() == ""):
        #             self.x1CoordEdit_11.setFocus()
        #             messageDisplay = "Cannot leave Baseline empty!"
        #             QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
        #             return

        if self.tabWidget_4.currentIndex()==1:

            if (self.wavelengthLe.text() == ""):
                self.wavelengthLe.setFocus()
                messageDisplay = "Cannot leave Wavelength empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return


        if self.tabWidget_4.currentIndex()==2:
            if self.tukeyRb.isChecked():

                if (self.x1CoordEdit_10.text() == ""):
                    self.x1CoordEdit_10.setFocus()
                    messageDisplay = "Cannot leave Wavelength empty!"
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
        try:
            self.inFile = self.lineEdit_2.text()
            self.outFile = self.lineEdit_3.text()
            self.outputFilename=self.outFile
            self.Metafile = self.lineEdit.text()

            print("In: " + self.inFile)
            print("Out: " + self.outFile)
            print("Meta data: " + self.Metafile)
            print("Running...")
            df_spectra = pd.read_csv(self.filepath,
                                     header=0, index_col=0)
            self.spectra = df_spectra.to_numpy()
            roi_ds = pd.read_csv(self.metafilepath, header=None, index_col=0)
            df = roi_ds.loc['class_label']

            dupl = df.duplicated()
            self.wavelength = df_spectra.index.values
            dupl_obs_samples = dupl[dupl == False].index.values
            self.no_days = np.shape(dupl_obs_samples)[0]
            self.days_emerg = roi_ds.loc['class_label'].values
            self.crop = roi_ds.loc['class'].values
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


            import seaborn as sns

            min_wv = np.min(self.wavelength)
            max_wv = np.max(self.wavelength)

            if self.plotAllMuStdRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.plotAllMeanStd()
                QApplication.restoreOverrideCursor()
            if self.plot2ClassesRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.plot2Classes()
                QApplication.restoreOverrideCursor()

            if self.plotContinuumRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.plotContinuum()
                QApplication.restoreOverrideCursor()

            if self.sdiRb.isChecked():
                self.plotSDI()
            if self.annovaRb.isChecked():
                self.OneWayAnova()
            if self.tukeyRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                QApplication.restoreOverrideCursor()

                in_value = int(self.x1CoordEdit_10.text())

                if in_value < min_wv or in_value > max_wv:
                    messageDisplay = "Invalid Wavelength Value. Value should be between " + str(min_wv) + " and " + str(
                        max_wv)
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                MultiComp=self.Tukey_multi_comparison()
                dlg = Dialog()
                dlg.textBrowser.append(MultiComp.tukeyhsd().summary().as_text())
                dlg.exec_()

            if self.kruskalRb.isChecked():
                self.Kruskal_Wallis()


            if self.plotDistnRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                in_value=int(self.wavelengthLe.text())

                if in_value<min_wv or in_value >max_wv:
                    messageDisplay = "Invalid Wavelength Value. Value should be between "+str(min_wv)+" and "+str(max_wv)
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                self.Band_plot()
                QApplication.restoreOverrideCursor()
            if self.plotCorrRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.Plot_Class_Correlate()
                QApplication.restoreOverrideCursor()

            if self.plotStatsRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.Plot_Statistics()
                QApplication.restoreOverrideCursor()

            if self.plotOneParamRb.isChecked():
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.plotSingleparameter()
                QApplication.restoreOverrideCursor()

            if self.plotAllParamRb.isChecked():
                self.parameters= roi_ds.index.values
                self.meta_transpose = roi_ds.T
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                self.plotMultipleparameter()
                QApplication.restoreOverrideCursor()
            print("Completed!!!")
        except Exception as e:
            import traceback
            print(e, traceback.format_exc())
            QApplication.restoreOverrideCursor()
            # print(e)




if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    # QSizePolicy sretain=Form.sizePolicy()
    # sretain.setRetainSizeWhenHidden(True)
    # sretain.setSizePolicy()
    ui = Visualizer()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
