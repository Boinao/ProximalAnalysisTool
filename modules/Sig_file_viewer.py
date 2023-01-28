# -*- coding: utf-8 -*-
"""
Created on Fri december 11 14:05:06 2020

@author: Nidhin
"""
from matplotlib.ticker import FormatStrFormatter
import collections
import math
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import specdal
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.SigreaderUi import Ui_Form
import os
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from modules.PandasModel import PandasModel
import re
import matplotlib.pyplot as plt

pluginPath = os.path.split(os.path.dirname(__file__))[0]

POSTFIX = '_SigReader'
from os import path


class Sigreader(Ui_Form):

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
        super(Sigreader, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveas())
        self.pushButton_4.clicked.connect(lambda: self.saveTable(self.data))
        self.pushButton_3.clicked.connect(lambda: self.getInputsrfFileName())
        self.pushButton_5.clicked.connect(lambda: self.getInputHDRFileName())
        self.OTHER_2.toggled.connect(lambda: self.toggleRadio_OTHER)
        self.AVIRISNG_2.toggled.connect(self.toggleRadio_AVIRISNG)
        self.radioButton_bbl.toggled.connect(lambda: self.toggleRadio_bbl)

    def toggleRadio_OTHER(self):
        self.pushButton_3.setEnabled(self.OTHER_2.isChecked())

    def toggleRadio_AVIRISNG(self):
        self.radioButton_bbl.setEnabled(self.AVIRISNG_2.isChecked())

    def toggleRadio_bbl(self):
        self.lineEdit_4.setEnabled(self.radioButton_bbl.isChecked())

    def getInputsrfFileName(self):
        return self.fillInputsrfFileEdit()

    def getInputHDRFileName(self):
        return self.fillInputbblFileEdit()

    # def getOutputFileName(self):
    #     return self.outSelector.filename()

    def fillInputsrfFileEdit(self):
        if self.OTHER_2.isChecked():
            self.inputFile, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.xlsx)",
                                                            directory=self.curdir)
            if (len(self.inputFile) == 0):
                messageDisplay = "Select atleast 1 file!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            OTHER_2filepath = self.inputFile
            self.lineEdit_3.setText(self.inputFile)
            return OTHER_2filepath

    def fillInputbblFileEdit(self):
        if self.AVIRISNG_2 and self.radioButton_bbl.isChecked():
            self.inputbblFile, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.hdr)",
                                                               directory=self.curdir)
            if (len(self.inputbblFile) == 0):
                messageDisplay = "Select atleast 1 file!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            HDRfilepath = self.inputbblFile
            self.lineEdit_4.setText(self.inputbblFile)
            return HDRfilepath

    def selectOperation(self):
        if (self.RAW_2.isChecked() == True):
            self.setTable("raw")
        elif (self.REFL_2.isChecked() == True):
            self.setTable("reflectance")
        elif (self.RAD_2.isChecked() == True):
            self.setTable("radiance")
        elif (self.REF_2.isChecked() == True):
            self.setTable("white_reference")

        elif (self.RAW_2.isChecked() == True):
            if (self.RAW_2.isChecked() == True):
                self.plotGraph("raw")
            elif (self.REFL_2.isChecked() == True):
                self.plotGraph("reflectance")
            elif (self.RAD_2.isChecked() == True):
                self.plotGraph("radiance")
            elif (self.REF_2.isChecked() == True):
                self.plotGraph("white_reference")

        else:
            messageDisplay = "Select one option to proceed!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)

    #        else:

    def browseButton_clicked(self):
        fname = []
        if self.curdir is None:
            self.curdir = os.getcwd()
            self.curdir = self.curdir.replace("\\", "/")

        self.lineEdit_2.setText("")
        fname, _ = QFileDialog.getOpenFileNames(None, filter="Supported types (*.sig)", directory=self.curdir)
        if (len(fname) == 0):
            messageDisplay = "Select atleast 1 file!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        self.filepath = fname
        if fname:
            inputText = str(fname[0]) + " "
            for i in range(1, len(fname)):
                inputText = inputText + " " + fname[i]
            self.lineEdit_2.setText(inputText)
            for i in range(0, len(fname)):
                item = str(fname[i].split('/')[-1].split('.')[0])
                self.listWidget.addItem(item)
            self.outputFilename = (str(self.curdir)) + "/Output" + POSTFIX + ".csv"
            self.lineEdit.setText(self.outputFilename)
        else:
            self.lineEdit_2.setText("")

    #    def getdataframe(self,filename):
    #        df=pd.DataFrame()
    #        df2=pd.DataFrame()
    #        df2.index.name='wavelength'
    #        for i in filename:
    #            df1=df.append(self.readASD(i)).transpose()
    #            df1.index.name='wavelength'
    #            df2=df2.merge(df1,how='outer',left_index=True,right_index=True)
    #        return df2
    #
    #    def readASD(self,filename):
    #        s=Spectrum(filepath=filename)
    #        #print(s.measurement)
    #        df=s.measurement
    #        return df

    def readASDclass(self, filename, function):
        df = pd.read_csv(filename, error_bad_lines=False)
        if (self.REFL_2.isChecked() == True):
            if len(df) == 516:

                data = pd.DataFrame(df)
                data.rename(columns={'///GER SIGNATUR FILE///': 'col2'}, inplace=True)
                # Drop rows
                data.drop(data.head(4).index, inplace=True)
                # data.rename(columns={'0': 'col2'}, inplace=True)
                data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
                reference = data.col2.str.split(' ', expand=True)[2]
                target = data.col2.str.split(' ', expand=True)[4]
                data['wavelength'] = data['wavelength'].astype('float')
                reference = reference.astype('float')
                target = target.astype('float')

                # data.reset_index(drop=True, inplace=True)
                # data.index.name = 'Index'
                data['Reflectance'] = target / reference
                data.drop(['col2'], axis=1, inplace=True)
                data.index = data.wavelength
                # Remove column name
                data.drop(['wavelength'], axis=1, inplace=True)
                return data
                # print(data)
            else:
                data = pd.DataFrame(df)
                data.rename(columns={'/*** Spectra Vista SIG Data ***/': 'col2'}, inplace=True)
                # Drop rows
                data.drop(data.head(5).index, inplace=True)
                # data.rename(columns={'0': 'col2'}, inplace=True)
                # print(data)
                data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
                reference = data.col2.str.split(' ', expand=True)[2]
                target = data.col2.str.split(' ', expand=True)[4]
                Reflectance = data.col2.str.split(' ', expand=True)[6]
                # print(data)
                # data['wavelength'] = data['wavelength'].astype('float')
                # data['reference'] = data['reference'].astype('float')
                # data['target'] = data['target'].astype('float')
                Reflectance = Reflectance.astype('float')
                data['Reflectance'] = Reflectance / 100
                data.drop(['col2'], axis=1, inplace=True)
                # print(data)
                data.index = data.wavelength
                # Remove column name
                data.drop(['wavelength'], axis=1, inplace=True)
                return data
        elif (self.REF_2.isChecked() == True):
            if len(df) == 516:

                data = pd.DataFrame(df)
                data.rename(columns={'///GER SIGNATUR FILE///': 'col2'}, inplace=True)
                # Drop rows
                data.drop(data.head(4).index, inplace=True)
                # data.rename(columns={'0': 'col2'}, inplace=True)
                data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
                reference = data.col2.str.split(' ', expand=True)[2]
                target = data.col2.str.split(' ', expand=True)[4]
                data['wavelength'] = data['wavelength'].astype('float')
                reference = reference.astype('float')
                target = target.astype('float')

                # data.reset_index(drop=True, inplace=True)
                # data.index.name = 'Index'
                # data['Reflectance'] = target / reference
                data['Reference'] = reference
                data.drop(['col2'], axis=1, inplace=True)
                data.index = data.wavelength
                # Remove column name
                data.drop(['wavelength'], axis=1, inplace=True)
                return data
            else:
                data = pd.DataFrame(df)
                data.rename(columns={'/*** Spectra Vista SIG Data ***/': 'col2'}, inplace=True)
                # Drop rows
                data.drop(data.head(5).index, inplace=True)
                # data.rename(columns={'0': 'col2'}, inplace=True)
                data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
                reference = data.col2.str.split(' ', expand=True)[2]
                # print(reference)
                target = data.col2.str.split(' ', expand=True)[4]
                data['wavelength'] = data['wavelength'].astype('float')
                reference = reference.astype('float')
                target = target.astype('float')
                data['Reference'] = reference / 100
                data.drop(['col2'], axis=1, inplace=True)
                data.index = data.wavelength
                # Remove column name
                data.drop(['wavelength'], axis=1, inplace=True)
                return data
        elif (self.RAD_2.isChecked() == True):
            if len(df) == 516:
                raise TypeError('spectral data contains {}. RAD data is needed')
            else:
                raise TypeError('spectral data contains {}. RAD data is needed')
        elif (self.RAW_2.isChecked() == True):
            if len(df) == 516:
                raise TypeError('old svc time')
            else:
                raise TypeError('new svc time')

    def plotGraph(self, df2, operation):
        ymax = max(df2.max(axis=0, skipna=True))
        ymin = min(df2.min(axis=0, skipna=True))
        if (operation == 'raw'):
            ylabel = 'DN'
        else:
            ylabel = operation

        if (operation == 'reflectance'):
            ymax = 1.25
            ymin = 0.0
        else:
            ymax = ymax
            ymin = ymin

        for i in range(0, len(self.filepath)):
            label = (self.filepath[i].split('/')[-1].split('.')[0])
            self.plot(df2.index, df2.iloc[:, i], label, ylabel, ymin, ymax)
        return df2

    def prepare_data(self, operation, jumpcor):

        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df2.index.name = 'wavelength'
        for i in self.filepath:
            df1 = df.append(self.readASDclass(i, operation))
            df1.index.name = 'wavelength'
            df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)
        samples, cols = df2.shape
        spectra = np.array(df2.iloc[0:samples + 1, 0:cols + 1])  # read Spectra
        # wavelength = list(np.floor(float(df2[0])))
        # Index column to float
        copy = df2.copy()
        x = copy.index.astype(float)
        # Index column to Integer
        y = x.astype(int)
        wavelength = list(y)
        # wavelength = list(df2.index)
        # wavelength = list(np.arange(290, 1095))
        # self.mplWidget_2.ax.set_xticks(wavelength[::100])# read wavelength

        # '''----------- Remove offset Error-------------'''
        # if jumpcor == True:
        #     S = spectra.copy()
        #     # SS = np.zeros((row, col))
        #     SS = S
        #     a = S[350, :] - S[351, :]
        #     b = S[351::, :]
        #     r, t = b.shape
        #     import numpy.matlib
        #     rep = np.matlib.repmat(a, r, 1)
        #     SS[351::, :] = rep + b
        #     a = SS[480, :] - SS[481, :]
        #     b = S[481:, :]
        #     r, t = b.shape
        #     rep = np.matlib.repmat(a, r, 1)
        #     SS[481:, :] = rep + b
        #     spectra = SS
        # else:
        #     spectra = spectra
        ##########################################################################
        filepaths = []
        for i in range(0, len(self.filepath)):
            filepaths.append(str(self.filepath[i].split('/')[-1].split('.')[0]))
        df2 = pd.DataFrame(spectra, index=wavelength, columns=filepaths)
        return df2

        # filepaths = []
        # for i in range(0, len(self.filepath)):
        #     filepaths.append(str(self.filepath[i].split('/')[-1].split('.')[0]))
        # df2.columns = filepaths
        # #        print(df2)
        # # fig, ax = self.mplWidget_2.ax.subplot(figsize=(10, 5))
        # for i in range(0, len(self.filepath)):
        #     label = (self.filepath[i].split('/')[-1].split('.')[0])
        #     self.mplWidget_2.ax.plot(df2.iloc[:, i], label=label)
        #     self.mplWidget_2.ax.legend()
        #     self.mplWidget_2.canvas.figure.autofmt_xdate()
        #     self.mplWidget_2.canvas.draw()
        # xtick = list(df2.index)
        # self.mplWidget_2.ax.set_xticks(xtick[::100])
        # self.mplWidget_2.setXAxisCaption("wavelength")
        # if (operation == 'Reflectance'):
        #     self.mplWidget_2.setYAxisCaption('Reflectance')
        # else:
        #     self.mplWidget_2.setYAxisCaption(operation)

    #        plt.plot(df2)
    #        plt.xlabel("wavelength")
    #        plt.ylabel("DN")
    def others_resample(self, df2, operation, jumpcor, filepath1):
        SRF = pd.read_excel(filepath1, sheet_name='Sheet1')
        samples, cols = SRF.shape
        mult_wave = list(SRF.iloc[:, 0])
        SRF_meta = pd.read_excel(filepath1, sheet_name='Sheet0')  # ,index_col=0)
        samples, cols = SRF_meta.shape
        cw = list(SRF_meta.iloc[:, 2])
        n_bands = len(cw)

        samples, cols = df2.shape
        spectra = np.array(df2.iloc[0:samples + 1, 0:cols + 1])  # read Spectra
        wavelength = list(np.arange(336, 1076))  # read wavelength

        '''----------- Remove offset Error-------------'''
        if jumpcor == True:
            S = spectra.copy()
            # SS = np.zeros((row, col))
            SS = S
            a = S[350, :] - S[351, :]
            b = S[351::, :]
            r, t = b.shape
            import numpy.matlib
            rep = np.matlib.repmat(a, r, 1)
            SS[351::, :] = rep + b
            a = SS[480, :] - SS[481, :]
            b = S[481:, :]
            r, t = b.shape
            rep = np.matlib.repmat(a, r, 1)
            SS[481:, :] = rep + b
            spectra = SS
        else:
            spectra = spectra

        spectra = spectra.T
        samples, cols = spectra.shape

        ###############
        ymax = max(df2.max(axis=0, skipna=True))
        ymin = min(df2.min(axis=0, skipna=True))

        if (operation == 'raw'):
            ylabel = 'DN'
            ymax = ymax
            ymin = ymin
        elif (operation == 'reflectance'):
            ymax = 1.25
            ymin = 0.0
            ylabel = operation
        else:
            ymax = ymax
            ymin = ymin
            ylabel = operation

        if self.radioButton_bbl.isChecked():
            bbl = self.bad_band(self.inputbblFile)
            bands = list(cw[np.nonzero(bbl)[0]])
            aviris = aviris_asd[:, np.nonzero(bbl)[0]]
            lab = []
            for i in range(0, len(self.filepath)):
                label = (self.filepath[i].split('/')[-1].split('.')[0])
                self.plot(bands, aviris[i, :], label, ylabel, ymin, ymax)
                lab.append(label)
            df = pd.DataFrame(aviris, index=lab, columns=bands)
        else:
            lab = []
            for i in range(0, len(self.filepath)):
                label = (self.filepath[i].split('/')[-1].split('.')[0])
                self.plot(cw, aviris_asd[i, :], label, ylabel, ymin, ymax)
                lab.append(label)
            df = pd.DataFrame(aviris_asd, index=lab, columns=cw)

        return df

    def saveTable(self, df):
        model = self.tableView.model()
        #        print(model.columnCount())
        dfNew = pd.DataFrame()
        dfNew = df
        data = []
        for i in range(0, model.rowCount()):
            data.append([])
            for j in range(0, model.columnCount()):
                data[i].append(model.data(model.index(i, j)))
        dfNew = pd.DataFrame(data)
        dfNew.columns = data[0]
        dfNew = dfNew.drop(0)
        # print(df.T.index)
        dfNew.index = df.T.index[1:]
        # print(dfNew)
        dfNew.to_csv(self.outputFilename)

    def saveas(self):
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', self.curdir, '*.csv')
        if self.outputFilename:
            self.lineEdit.setText(self.outputFilename)

        return self.outputFilename

    def run(self):
        inFile = self.fillInputbblFileEdit()
        # self.outFile = self.getOutputFileName()
        srfFile = self.fillInputsrfFileEdit()
        if self.checkBox.isChecked():
            jumpcor = True
        else:
            jumpcor = False
        if (self.lineEdit_2.text() == ""):
            messageDisplay = "Cannot leave Input empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        filepath = str(self.lineEdit_2.text()).split()
        for i in filepath:
            if (str(i.split('/')[-1].split('.')[-1]) == "sig"):
                pass
            else:
                self.lineEdit_2.setFocus()
                self.lineEdit_2.selectAll()
                messageDisplay = "Input file extension cannot be " + str(i.split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
        #        print(filepath)
        c = 0
        count = 0
        # print(len(filepath))
        for i in range(len(filepath)):
            if (len(self.filepath) == 0):
                if (path.exists(filepath[i].rsplit('/', 1)[0]) == True):
                    pass
                else:
                    messageDisplay = "Path does not exist!"
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                if (path.isfile(filepath[i]) == True):
                    count = count + 1
                    pass
                else:
                    messageDisplay = "File does not Exist!"
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
            else:
                if (filepath[i] == self.filepath[i]):
                    continue
                else:
                    c = 1
                    break
        # print(count)
        # print(i)
        if (count == i + 1):
            self.filepath = filepath
        # print(self.filepath)

        if (c == 1):

            for i in filepath:
                # print(i.rsplit('/', 1)[0])
                # print(i)
                my_path = path.exists(i.rsplit('/', 1)[0])
                if (path.exists(i.rsplit('/', 1)[0]) == True):
                    pass
                else:
                    messageDisplay = "Path does not exist!"
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
                my_file = path.isfile(i)
                if (path.isfile(i) == True):
                    pass
                else:
                    messageDisplay = "File does not Exist!"
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return
            self.filepath = filepath
        self.listWidget.clear()
        for i in range(0, len(self.filepath)):
            item = str(self.filepath[i].split('/')[-1].split('.')[0])
            self.listWidget.addItem(item)
        if (self.lineEdit.text() == ""):
            self.lineEdit.setFocus()
            self.lineEdit.selectAll()
            messageDisplay = "Cannot leave Output empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        try:
            if self.REF_2.isChecked():
                self.mplWidget_2.clear()
                operation = 'white_reference'
                df2 = self.prepare_data(operation, jumpcor)
                self.plotGraph(df2, operation)

            elif self.REFL_2.isChecked():
                self.mplWidget_2.clear()
                operation = 'reflectance'
                df2 = self.prepare_data(operation, jumpcor)
                self.plotGraph(df2, operation)

            elif self.RAD_2.isChecked():
                self.mplWidget_2.clear()
                operation = 'radiance'
                # v = asd.reader(inFile)
                # ref = v.radiance
                df2 = self.prepare_data(operation, jumpcor)
                self.plotGraph(df2, operation)

            elif self.RAW_2.isChecked():
                self.mplWidget_2.clear()
                operation = 'raw'
                df2 = self.prepare_data(operation, jumpcor)
                self.plotGraph(df2, operation)


            QApplication.restoreOverrideCursor()
            # fileInfo = QFileInfo(self.outFile)
            # if fileInfo.exists():
            #     if self.checkBox_3.isChecked():
            #         self.addLayerIntoCanvas(fileInfo)
            #     QMessageBox.information(self, self.tr("Finished"), self.tr("Processing completed."))
            print('Completed!')

        except Exception as e:
            import traceback
            print(e, traceback.format_exc())
            QApplication.restoreOverrideCursor()
        outputPath = (self.lineEdit.text()).split()
        if (len(outputPath) == 1):
            pass
        else:
            messageDisplay = "Enter one ouput file only!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        # print(outputPath[0].rsplit('/', 1)[0])
        if (path.exists(outputPath[0].rsplit('/', 1)[0])):
            if (str(outputPath[0].split('/')[-1].split('.')[-1]) == "csv"):
                pass
            else:
                self.lineEdit.setFocus()
                self.lineEdit.selectAll()
                messageDisplay = "Output file extension cannot be " + str(outputPath[0].split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
        else:
            self.lineEdit.setFocus()
            self.lineEdit.selectAll()
            messageDisplay = "Output File Path does not exist!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        self.outputFilename = outputPath[0]
        self.selectOperation()
        df2.to_csv(self.outputFilename)
        del df2, operation
        # pass

    def plot(self, x, y, label, ylabel, ymin, ymax):
        self.mplWidget_2.ax.plot(x, y, label=label)
        self.mplWidget_2.ax.legend()
        self.mplWidget_2.canvas.figure.autofmt_xdate()
        self.mplWidget_2.canvas.draw()
        self.mplWidget_2.setTitle("SVC Spectroradiometer Data")
        self.mplWidget_2.setYAxisCaption(ylabel)
        self.mplWidget_2.ax.set_ylim(ymin, ymax)
        # self.mplWidget_2.ax.set_yticks(np.arange(ymin,ymax,int((ymax-ymin)/5)))
        self.mplWidget_2.setXAxisCaption("Wavelength (nm)")
        self.mplWidget_2.setYAxisCaption(ylabel)

    def setTable(self, operation):
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df2.index.name = 'wavelength'
        for i in self.filepath:
            df1 = df.append(self.readASDclass(i, operation))
            df1.index.name = 'wavelength'
            df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)
        #        print("1")
        filepaths = []
        for i in range(0, len(self.filepath)):
            filepaths.append(str(self.filepath[i].split('/')[-1].split('.')[0]))
        df2.columns = filepaths
        #        print("2")
        #        print(df2)
        #        df=self.readASDclass(self.filepath)
        #        print(df)
        #        filepaths=[]
        #        for i in range(0,len(self.filepath)):
        #            filepaths.append(str(self.filepath[i].split('/')[-1].split('.')[0]) )
        #        df.columns=filepaths
        #        plt.plot(df)
        #        print(df2.T)
        df2.T.to_csv(self.outputFilename)
        df2 = df2.reset_index()
        #        print("3")
        #        print(df2)
        self.data = df2
        #        print(df)
        self.model = PandasModel(df2.T)
        self.tableView.setModel(self.model)


''' Generate gaussian Response for specific wavelength'''


def gauss3(x, mu, fwhm):
    c2 = 2 * np.sqrt(2 * np.log(2))
    sigma = fwhm / c2
    c1 = -(np.square((x - mu) / sigma) * 0.5)
    return x, np.exp(c1)


''' generate gaussian for CW, FWHM'''


def srf_generate(center, fwhmm, num_samp, yedge):
    n2 = int(num_samp - 1) / 2
    n2 = int(n2)
    sam_idx = np.linspace(-n2, n2, num_samp)
    sigma = np.float64(np.sqrt(-np.power(np.float64(n2), 2.0) / 2.0 / np.log(yedge)))
    nfwhm = np.float64(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)
    y = np.float64(np.exp(-np.power(sam_idx, 2.0) / 2.0 / np.power(sigma, 2.0)))
    delta = np.float64(fwhmm / nfwhm)
    wl = np.float64((np.ones(num_samp) * center) + (delta * sam_idx))
    return wl, y


def read_hdr_file(hdrfilename, keep_case=False):
    """
    Read information from ENVI header file to a dictionary.
    By default all keys are converted to lowercase. To stop this behaviour
    and keep the origional case set 'keep_case = True'
    """
    ENVI_TO_NUMPY_DTYPE = {'1': np.uint8,
                           '2': np.int16,
                           '3': np.int32,
                           '4': np.float32,
                           '5': np.float64,
                           '6': np.complex64,
                           '9': np.complex128,
                           '12': np.uint16,
                           '13': np.uint32,
                           '14': np.int64,
                           '15': np.uint64}
    output = collections.OrderedDict()
    comments = ""
    inblock = False
    try:
        hdrfile = open(hdrfilename, "r")
    except Exception as e:
        print("Could not open hdr file " + str(hdrfilename) + \
                      ". Reason: " + e)
    # Read line, split it on equals, strip whitespace from resulting strings
    # and add key/value pair to output
    for currentline in hdrfile:
        # ENVI headers accept blocks bracketed by curly braces - check for these
        if not inblock:
            # Check for a comment
            if re.search("^;", currentline) is not None:
                comments += currentline
            # Split line on first equals sign
            elif re.search("=", currentline) is not None:
                linesplit = re.split("=", currentline, 1)
                key = linesplit[0].strip()
                # Convert all values to lower case unless requested to keep.
                if not keep_case:
                    key = key.lower()
                value = linesplit[1].strip()
                # If value starts with an open brace, it's the start of a block
                # - strip the brace off and read the rest of the block
                if re.match("{", value) is not None:
                    inblock = True
                    value = re.sub("^{", "", value, 1)
                    # If value ends with a close brace it's the end
                    # of the block as well - strip the brace off
                    if re.search("}$", value):
                        inblock = False
                        value = re.sub("}$", "", value, 1)
                value = value.strip()
                output[key] = value
        else:
            # If we're in a block, just read the line, strip whitespace
            # (and any closing brace ending the block) and add the whole thing
            value = currentline.strip()
            if re.search("}$", value):
                inblock = False
                value = re.sub("}$", "", value, 1)
                value = value.strip()
            output[key] = output[key] + value
    hdrfile.close()
    output['_comments'] = comments
    return output


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    ui = Sigreader()
    #    self.setFixedSize(self.layout.sizeHint())
    self.resize(minimumSizeHint())
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
