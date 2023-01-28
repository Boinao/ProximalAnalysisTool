# -*- coding: utf-8 -*-
"""
Created on Fri december 11 14:05:06 2020

@author: Nidhin
"""
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import specdal
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from Ui.SigreaderUi import Ui_Form
import os
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from modules.PandasModel import PandasModel

import matplotlib.pyplot as plt

POSTFIX = '_SigReader'
from os import path
from modules import Utils

class SigNewreader(Ui_Form):

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
        super(SigNewreader, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveas())
        self.pushButton_4.clicked.connect(lambda: self.saveTable(self.data))

    def selectOperation(self):
        if (self.radioButton_4.isChecked() == True):
            if (self.radioButton_2.isChecked() == True):
                self.setTable("reflectance")
            elif (self.radioButton_3.isChecked() == True):
                self.setTable("radiance")
            elif (self.radioButton_6.isChecked() == True):
                self.setTable("white_reference")
        elif (self.radioButton_5.isChecked() == True):
            if (self.radioButton_2.isChecked() == True):
                self.plotGraph("reflectance")
            elif (self.radioButton_3.isChecked() == True):
                self.plotGraph("radiance")
            elif (self.radioButton_6.isChecked() == True):
                self.plotGraph("white_reference")

        else:
            messageDisplay = "Select one option to proceed!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)

    #        else:

    def browseButton_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()
        self.lineEdit_2.setText("")
        self.listWidget.clear()

        fname, _ = QFileDialog.getOpenFileNames(None, filter="Supported types (*.sig)", directory=lastDataDir)

        if not fname:
            return
        self.filepath = fname

        if fname:
            directory = os.path.dirname(fname[0])
            if len(fname) > 0:
                # fname = "\"".join(fname, "\"," )
                fn = ";".join(fname)
            else:
                fn = fname[0]

            for files in fname:
                self.listWidget.addItem(os.path.basename(files).split('.')[0])
            self.lineEdit_2.setText(fn)
            self.outputFilename = directory + "/Output" + POSTFIX + ".csv"
            self.lineEdit.setText(self.outputFilename)
            Utils.setLastUsedDir(directory)
        else:
            self.lineEdit_2.setText("")


    def setTable(self, operation):
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df2.index.name = 'wavelength'
        for i in self.filepath:
            df1 = df.append(self.readASDclass(i, operation))
            df1.index.name = 'wavelength'
            df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)
        #        print("1")
        print(df2)
        filepaths = []
        # for i in range(0, len(self.filepath)):
        #     filepaths.append(str(self.filepath[i].split('/')[-1].split('.')[0]))
        df2.columns = self.filepath

        df2.T.to_csv(self.outputFilename)
        df2 = df2.reset_index()
        self.data = df2
        #        print(df)
        self.model = PandasModel(df2.T)
        self.tableView.setModel(self.model)

    def readASDclass(self, filename, function):
            df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False)
            if (self.radioButton_2.isChecked() == True):
                try:
                    data = pd.DataFrame(df)
                    print(data,"data")
                    data.rename(columns={'/*** Spectra Vista SIG Data ***/': 'col2'}, inplace=True)
                    # Drop rows
                    data.drop(data.head(5).index, inplace=True)
                    # data.rename(columns={'0': 'col2'}, inplace=True)
                    # print(data)
                    data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
                    print(data,"wavelength")
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
                    print(data)
                    data.index = data.wavelength
                    # Remove column name
                    data.drop(['wavelength'], axis=1, inplace=True)
                    return data
                except Exception as e:
                    messageDisplay = "Error :"+str(e)
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            elif (self.radioButton_6.isChecked() == True):
                try:
                    data = pd.DataFrame(df)
                    print(data)
                    data.rename(columns={'/*** Spectra Vista SIG Data ***/': 'col2'}, inplace=True)
                    # Drop rows
                    data.drop(data.head(5).index, inplace=True)
                    # data.rename(columns={'0': 'col2'}, inplace=True)
                    data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
                    reference = data.col2.str.split(' ', expand=True)[2]
                    print(reference)
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
                except Exception as e:
                    messageDisplay = "Error :"+str(e)
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            elif (self.radioButton_3.isChecked() == True):
                try:
                    data = pd.DataFrame(df)
                    print(data)
                    data.rename(columns={'/*** Spectra Vista SIG Data ***/': 'col2'}, inplace=True)
                    # Drop rows
                    data.drop(data.head(5).index, inplace=True)
                    # data.rename(columns={'0': 'col2'}, inplace=True)
                    data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
                    reference = data.col2.str.split(' ', expand=True)[2]
                    print(reference)
                    target = data.col2.str.split(' ', expand=True)[4]
                    data['wavelength'] = data['wavelength'].astype('float')
                    reference = reference.astype('float')
                    data['target'] = target.astype('float')
                    data.drop(['col2'], axis=1, inplace=True)
                    data.index = data.wavelength
                    # Remove column name
                    data.drop(['wavelength'], axis=1, inplace=True)
                    return data
                except Exception as e:
                    messageDisplay = "Error :"+str(e)
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)




    def plotGraph(self, operation):
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df2.index.name = 'wavelength'
        for i in self.filepath:
            df1 = df.append(self.readASDclass(i, operation))
            df1.index.name = 'wavelength'
            df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)
        filepaths = []
        for i in range(0, len(self.filepath)):
            filepaths.append(str(self.filepath[i].split('/')[-1].split('.')[0]))
        df2.columns = filepaths
        #        print(df2)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(0, len(self.filepath)):
            label = (self.filepath[i].split('/')[-1].split('.')[0])
            ax.plot(df2.iloc[:, i], label=label)
            ax.legend()
        xtick = list(df2.index)
        ax.set_xticks(xtick[::100])
        plt.xlabel("wavelength")
        if (operation == 'reflectance'):
            plt.ylabel('Reflectance')
        else:
            plt.ylabel(operation)

    #        plt.plot(df2)
    #        plt.xlabel("wavelength")
    #        plt.ylabel("DN")

    def saveTable(self, df):


        model = self.tableView.model()
        if model is None:
            messageDisplay = "Empty data cannot be saved!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (model.rowCount() == 0):
            messageDisplay = "Empty data cannot be saved!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

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

        dfNew.index = df.T.index[1:]
        dfNew.T.to_csv(self.outputFilename)

    def saveas(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')
        if not self.outputFilename:
            return

        self.lineEdit.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename

    def run(self):
        if (self.lineEdit_2.text() is None) or (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            messageDisplay = "Cannot leave Input empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
            self.lineEdit.setFocus()
            messageDisplay = "Cannot leave Output empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.isdir(os.path.dirname(self.lineEdit.text()))):
            self.lineEdit.setFocus()
            QtWidgets.QMessageBox.information(self.Form, 'Error', "Kindly enter a valid output path.",
                                              QtWidgets.QMessageBox.Ok)
            return

        if not str(self.lineEdit.text()).split(".")[-1] == 'csv':
            self.lineEdit.setFocus()
            messageDisplay = "Output file extension cannot be " + str(self.lineEdit_2.text()).split(".")[-1]
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        filepath = str(self.lineEdit_2.text()).split(";")
        print(filepath)

        for file in filepath:
            if (not os.path.exists(file)):
                messageDisplay = "Path does not exist : " + file
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if not str(os.path.basename(file).split('.')[-1]) == "sig":
                messageDisplay = "Input file extension cannot be " + str(os.path.basename().split('.')[-1])
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        self.outputFilename = self.lineEdit.text()
        self.selectOperation()
        # pass


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    ui = SigNewreader()
    #    self.setFixedSize(self.layout.sizeHint())
    self.resize(minimumSizeHint())
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
