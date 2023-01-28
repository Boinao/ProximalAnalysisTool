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
from PyQt5.QtWidgets import QWidget
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
    compareCmbLoaded = False
    compareBandSepLoaded = False
    compareStatistics = False
    statsSingle = False

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

    def setupUi(self, Form, kind):
        super(Visualizer, self).setupUi(Form)
        self.Form = Form
        self.kind = kind

        # print(self.tabWidget_4.count())

        if self.kind==TIME_SERIES:
            while self.tabWidget_4.count()>3:
                self.tabWidget_4.removeTab(self.tabWidget_4.count()-1)

        # print(self.tabWidget_4.count())

        self.baselineLe.setText("0.0")
        self.baselineLe.setVisible(False)
        self.label_73.setVisible(False)

        self.connectWidgets()
        self.metafilepath = None
        self.cb=None


    def connectWidgets(self):
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_5.clicked.connect(lambda: self.MetadatabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())
        self.plot2ClassesRb.toggled.connect(self.toggleRadio_rb_sep_compare)
        self.sdiRb.toggled.connect(self.toggleSDI)
        self.annovaRb.toggled.connect(self.toggleAnnova)
        self.tukeyRb.toggled.connect(self.toggleTukey)
        self.kruskalRb.toggled.connect(self.toggleKruskal)
        self.plotDistnRb.toggled.connect(self.toggleDistn)
        self.plotStatsRb.toggled.connect(self.toggleRadioStatistics)
        self.plotOneParamRb.toggled.connect(self.toggleRadioSingle)

        # self.x1CoordEdit_10.setValidator(QIntValidator())
        # self.wavelengthLe.setValidator(QIntValidator())

    def clearButtonGrp(self):
        self.buttonGroup.setExclusive(False)
        self.buttonGroup.checkedButton().setChecked(False)
        self.buttonGroup.setExclusive(True)

    def checkMetadata(self):
        self.metafilepath=self.lineEdit.text()
        if (self.metafilepath is None ) or (self.metafilepath==''):
            # print('Input data or metadatafile may be empty')
            QtWidgets.QMessageBox.information(self.Form, 'Message', 'Input data or metadatafile may be empty',
                                              QtWidgets.QMessageBox.Ok)
            self.clearButtonGrp()
            return True
        return False

    def read_metadata(self):
        kind=self.kind
        self.checkMetadata()

        metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
        if kind==NORMAL:
            classes = metadata.loc['class_label'].unique().astype(int)
        elif kind==TIME_SERIES:
            classes = metadata.loc['Days'].unique().astype(int)
        return classes

    def populateCombobox(self,srcCombobox, tgtCombobox, classes):
        srcCombobox.clear()
        tgtCombobox.clear()
        srcCombobox.addItem("--Select--")
        kind = self.kind
        for i in classes:
            if kind==NORMAL:
                srcCombobox.addItem("Class " + str(i))
            elif kind==TIME_SERIES:
                srcCombobox.addItem("Day " + str(i))

    def toggleRadio_rb_sep_compare(self):
        self.comboBox_25.setEnabled(self.plot2ClassesRb.isChecked())
        self.comboBox_26.setEnabled(self.plot2ClassesRb.isChecked())
        if self.plot2ClassesRb.isChecked():
            if self.checkMetadata():
                return

            if not Visualizer.compareCmbLoaded:

                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                classes=self.read_metadata()
                QApplication.restoreOverrideCursor()

                self.populateCombobox(self.comboBox_25,self.comboBox_26, classes)
                self.roi_ds=classes
                self.comboBox_25.currentIndexChanged.connect(self.updateSecondCombo)
                Visualizer.compareCmbLoaded=True



    def loadDependentCmb(self, sourceRb):

        self.comboBox_classA_8.setEnabled(sourceRb.isChecked())
        self.comboBox_classB_8.setEnabled(sourceRb.isChecked())

        if self.sdiRb.isChecked() or self.annovaRb.isChecked() or self.kruskalRb.isChecked():
            if self.checkMetadata():
                return

            if not Visualizer.compareBandSepLoaded:
                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                classes = self.read_metadata()
                QApplication.restoreOverrideCursor()

                self.populateCombobox(self.comboBox_classA_8, self.comboBox_classB_8,classes)

                self.SDIsecond = classes
                self.comboBox_classA_8.currentIndexChanged.connect(self.SDISecondCombo)
                Visualizer.compareBandSepLoaded = True


    def toggleSDI(self):
        self.loadDependentCmb(self.sdiRb)

    def toggleAnnova(self):
        self.loadDependentCmb(self.annovaRb)

    def toggleKruskal(self):
        self.loadDependentCmb(self.kruskalRb)

    def toggleTukey(self):
        self.wavelengthCmb2.setEnabled(self.tukeyRb.isChecked())
        # self.x1CoordEdit_10.setEnabled(self.tukeyRb.isChecked())

    def toggleRadio_pairplot(self):
        self.comboBox_29.setEnabled(self.radioPairplot_5.isChecked())
        self.comboBox_30.setEnabled(self.radioPairplot_5.isChecked())

    def toggleDistn(self):
        self.wavelengthCmb.setEnabled(self.plotDistnRb.isChecked())


    # def onLayersChanged(self):
    #     self.inSelector.setLayers(Utils.LayerRegistry.instance().getRasterLayers())

    def loadStatsCmb(self, srcCmb,sourceRb):
        srcCmb.setEnabled(sourceRb.isChecked())
        if sourceRb.isChecked():
            if self.checkMetadata():
                return

            self.meta_transpose = None
            metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)

            srcCmb.clear()
            srcCmb.addItem("--Select")
            meta_copy = metadata.T
            self.meta_transpose = metadata.T
            # self.meta_transpose.drop(columns=['class'], axis=1, inplace=True)
            meta_copy.drop(["Spectra ID", "class_label", "class"], axis=1, inplace=True)
            for i in meta_copy.columns:
                srcCmb.addItem(str(i))



    def toggleRadioStatistics(self):
        self.comboBox_31.setEnabled(self.plotStatsRb.isChecked())
        if self.comboBox_31.count()<=0:
            self.loadStatsCmb(self.comboBox_31, self.plotStatsRb)
            if self.comboBox_31.count() > 1:
                Visualizer.compareStatistics=True


    def toggleRadioSingle(self):
        self.comboBox_33.setEnabled(self.plotOneParamRb.isChecked())
        if self.comboBox_33.count()<=0:
            self.loadStatsCmb(self.comboBox_33, self.plotOneParamRb)
            if self.comboBox_33.count() > 1:
                Visualizer.statsSingle=True


    def stateChanged1(self):
        self.lineEdit.setEnabled(True)

    def stateChanged2(self):
        self.lineEdit.setEnabled(True)

    def flushFlag(self):

        Visualizer.compareCmbLoaded = False
        Visualizer.compareBandSepLoaded = False
        Visualizer.compareStatistics = False
        Visualizer.statsSingle=False

    def SpectrabrowseButton_clicked(self):
        self.flushFlag()
        fname=Utils.browseInputFile(POSTFIX + ".csv", self.lineEdit_2, "Supported types (*.csv)", self.lineEdit_3)

        # print(self.filepath)
        if fname:
            if not Utils.validateInputFormat(fname):
                self.lineEdit_2.setText("")
                self.lineEdit_3.setText("")
                return

            Utils.populateWavelength(self.wavelengthCmb, self.lineEdit_2.text())
            Utils.populateWavelength(self.wavelengthCmb2, self.lineEdit_2.text())

        else:
            self.lineEdit_2.setText("")
            self.lineEdit_3.setText("")
            self.wavelengthCmb.clear()
            self.wavelengthCmb2.clear()



    def MetadatabrowseButton_clicked(self):
        self.flushFlag()
        fname=Utils.browseMetadataFile(self.lineEdit, "Supported types (*.csv)")

        if fname:
            if not Utils.validateMetaFormat(self.lineEdit_2.text(), fname):
                self.lineEdit.setText("")
                return


    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')
        if not self.outputFilename:
            return
        self.lineEdit_3.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename

    def saveas(self):
        self.outputFilename = Utils.browseSaveFile(self.lineEdit_3, '*.csv')


    def readData(self):
        df_spectra = pd.read_csv(self.inFile, header=0, index_col=0)
        df_metadata = pd.read_csv(self.Metafile, header=None, index_col=0)

        spectra = df_spectra.to_numpy()

        class_days = None
        labelTxt = ''
        class_name=''
        if self.kind == NORMAL:
            class_days = df_metadata.loc['class_label'].values.astype(np.int8)
            class_name = list(set(df_metadata.loc['class'].values.tolist()))
            labelTxt = 'Class'
        elif self.kind == TIME_SERIES:
            class_days = df_metadata.loc['Days'].values.astype(np.int8)
            class_name=np.unique(class_days)
            labelTxt = 'Days'


        return spectra,class_days,class_name, labelTxt

    def plotAllMeanStd(self):
        self.mplWidgetSpectral_5.clear()
        spectra,class_days,class_name, labelTxt=self.readData()

        unique_key = np.unique(class_days)
        avg_all_spectra = []
        std_all_spectra = []
        for index,i in enumerate(unique_key):
            mean_spectra = spectra[:, np.where(class_days == i)[0]].mean(axis=1)
            std_spectra = spectra[:, np.where(class_days == i)[0]].std(axis=1)

            lower = mean_spectra - std_spectra
            upper = mean_spectra + std_spectra

            self.mplWidgetSpectral_5.ax.plot(self.wavelength, mean_spectra,
                                             label=labelTxt+" " +str(index)+' = ' + str(class_name[index]),
                                             linewidth=3)
            # self.mplWidgetSpectral_5.ax.legend()
            legend = self.mplWidgetSpectral_5.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            legend.set_draggable(True)
            self.mplWidgetSpectral_5.ax.set_ylabel('Reflectance')
            self.mplWidgetSpectral_5.ax.set_xlabel('Wavelength')
            self.mplWidgetSpectral_5.ax.fill_between(self.wavelength, lower, upper, alpha=0.5)
            avg_all_spectra.append(mean_spectra)
            std_all_spectra.append(std_spectra)

        avg_all_spectra = np.asarray(avg_all_spectra).T
        std_all_spectra = np.asarray(std_all_spectra).T

        self.mplWidgetSpectral_5.canvas.draw()
        # print(avg_spectra_.shape,std_spectra_.shape)
        result = np.hstack((avg_all_spectra, std_all_spectra))
        # print(result.shape)
        df1 = pd.DataFrame(result, index=self.wavelength)
        # print([str(int(days_emerg[dupl_obs_samples[i]])) for i in range(no_days)]*2)
        head = [labelTxt +'_Mean_' + str(i) for i in unique_key] + [labelTxt+'_Std_' + str(i) for i in unique_key]
        df1.to_csv(self.outputFilename + '_Spectra' + '.csv', header=head, index=True)

    def plot2Classes(self):
        self.mplWidgetSpectral_5.clear()

        if self.comboBox_25.currentIndex()<1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first '+self.labelTxt+' A', QtWidgets.QMessageBox.Ok)
            self.comboBox_25.setFocus()
            return

        if self.comboBox_26.currentIndex()<1:
            QtWidgets.QMessageBox.information(self.Form, 'Error', 'Please select first '+self.labelTxt+' B', QtWidgets.QMessageBox.Ok)
            self.comboBox_26.setFocus()
            return


        label_one = int(self.comboBox_25.currentText().split()[1])
        label_two = int(self.comboBox_26.currentText().split()[1])
        spectra1 = self.df_mean[label_one]
        spectra2 = self.df_mean[label_two]
        # class_days=df_mean.columns.values
        self.mplWidgetSpectral_5.ax.plot(self.wavelength, spectra1, label=self.labelTxt+' = ' + str(label_one),
                                         # label='Class = ' + str(label_one),
                                         linewidth=3)
        self.mplWidgetSpectral_5.ax.plot(self.wavelength, spectra2, label=self.labelTxt+' = ' + str(label_two),
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
        baseline = float(self.baselineLe.text())
        spectra, class_days, class_name, labelTxt = self.readData()

        unique_key = np.unique(class_days)
        for v,i in enumerate(unique_key):
            spectra1 = spectra[:, np.where(class_days == i)[0]].mean(axis=1).tolist()
            wvl = self.wavelength.tolist()

            fea = spectro.FeaturesConvexHullQuotient(spectra1, wvl, baseline=baseline)
            self.mplWidgetSpectral_5.ax.plot(self.wavelength, fea.crs,
                                             label=labelTxt+' = ' + str(class_name[v]))
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
        kind = self.kind
        for i in second_labels:
            if kind==NORMAL:
                self.comboBox_26.addItem("Class " + str(i))
            elif kind==TIME_SERIES:
                self.comboBox_26.addItem("Day " + str(i))


    def SDISecondCombo(self, i):
        labels = self.SDIsecond
        # labels = range(0, labels)
        # second_labels = [x for x in labels if x != i - 1]
        second_labels = [x for x in labels if x != labels[i - 1]]
        # second_labels = [x for x in labels]
        self.mplWidgetSep.ax.clear()
        self.comboBox_classB_8.clear()
        self.comboBox_classB_8.addItem("--Select--")
        kind = self.kind
        for i in second_labels:
            if kind==NORMAL:
                self.comboBox_classB_8.addItem("Class " + str(i))
            elif kind==TIME_SERIES:
                self.comboBox_classB_8.addItem("Day " + str(i))

    def Band_plot(self):

        self.mplWidgetDist.ax.clear()
        test_wave = int(self.wavelengthCmb.currentText())
        # if i < 1:
        #     return
        # label_one = str(int(self.comboBox_3.currentText().split()[1]))
        df = pd.DataFrame()
        index = self.wavelength

        spectra, class_days, class_name, labelTxt = self.readData()

        unique_key = np.unique(class_days)
        for v,i in enumerate(unique_key):
            spectra1 = spectra[:, np.where(class_days == i)[0]]
            spectra1_dataframe = pd.DataFrame(spectra1, index=index)
            df[' ' + str(class_name[v])] = spectra1_dataframe.loc[test_wave]
        dataframe = pd.melt(df)
        # print(dataframe)
        dataframe.columns = [labelTxt, 'Values']
        sns.boxplot(x=labelTxt, y="Values", data=dataframe, ax=self.mplWidgetDist.ax)
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
        label_one = int(self.comboBox_classA_8.currentText().split()[1])
        label_two = int(self.comboBox_classB_8.currentText().split()[1])
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
        label_one = int(self.comboBox_classA_8.currentText().split()[1])
        label_two = int(self.comboBox_classB_8.currentText().split()[1])
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
        # test_wave = int(self.x1CoordEdit_10.text())
        test_wave = int(self.wavelengthCmb2.currentText())

        df = pd.DataFrame()
        index = self.wavelength

        spectra, class_days, class_name, labelTxt = self.readData()
        unique_key = np.unique(class_days)
        for v, i in enumerate(unique_key):
            spectra1 = spectra[:, np.where(class_days == i)[0]]
            spectra1_dataframe = pd.DataFrame(spectra1, index=index)
            df['day' + str(i)] = spectra1_dataframe.loc[test_wave]

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

        label_one = int(self.comboBox_classA_8.currentText().split()[1])
        label_two = int(self.comboBox_classB_8.currentText().split()[1])

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
        # if i < 1:
        #     return
        label_one = int(self.comboBox_classA_8.currentText().split()[1])
        label_two = int(self.comboBox_classB_8.currentText().split()[1])

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
        self.widget_corr_5.figure.clear()
        df_spectra = pd.read_csv(self.inFile,
                                 header=0, index_col=0)

        indxs = df_spectra.T.shape
        # # Compute the correlation matrix
        corr = df_spectra.T.corr()
        # print(corr)
        ax = self.widget_corr_5.figure.add_subplot(111)
        ax.set_aspect('auto')
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        # m=ax.imshow(corr,cmap='jet', interpolation='nearest')
        sns.heatmap(corr, cmap='RdYlGn', vmax=1,
                    xticklabels=max(5, indxs[1] // 10),
                    yticklabels=max(5, indxs[1] // 10),
                    square=False, cbar=True, mask=mask, ax=ax)


        ax.set_title("Wavelength Correlation")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Wavelength")

        # self.cb=self.widget_corr_5.figure.colorbar(m,ax=ax)
        # self.widget_corr_5.figure.subplots_adjust(right=0.70)

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

        if self.tabWidget_4.currentIndex()==1 and (self.wavelengthCmb.currentText() == ""):
            self.wavelengthCmb.setFocus()
            messageDisplay = "Cannot leave Wavelength empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return


        if self.tabWidget_4.currentIndex()==2 and self.tukeyRb.isChecked() and (self.wavelengthCmb2.currentText() == ""):
            self.wavelengthCmb2.setFocus()
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

            df_spectra = pd.read_csv(self.inFile, header=0, index_col=0)
            df_metadata = pd.read_csv(self.Metafile, header=None, index_col=0)
            self.parameters = df_metadata.index.values
            self.meta_transpose = df_metadata.T

            spectra = df_spectra.to_numpy()
            self.wavelength = df_spectra.index.values

            class_days = None
            self.labelTxt = ''
            if self.kind == NORMAL:
                class_days = df_metadata.loc['class_label'].values.astype(np.int8)
                self.labelTxt = 'Class'
            elif self.kind == TIME_SERIES:
                class_days = df_metadata.loc['Days'].values.astype(np.int8)
                self.labelTxt = 'Day'

            unique_key = np.unique(class_days)
            self.df_mean = pd.DataFrame()
            self.df_std = pd.DataFrame()
            self.df = pd.DataFrame()
            for i in unique_key:
                self.df_mean[i] = spectra[:, np.where(class_days == i)[0]].mean(axis=1)
                self.df_std[i] = spectra[:, np.where(class_days == i)[0]].std(axis=1)
                self.df[i] = spectra[:, np.where(class_days == i)[0]].tolist()



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

                in_value = int(self.wavelengthCmb2.currentText())

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
                in_value=int(self.wavelengthCmb.currentText())

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
    ui = Visualizer()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
