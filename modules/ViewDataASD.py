# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:18:02 2019

@author: Trainee
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:05:06 2019

@author: Trainee
"""
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import specdal
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.ViewDataUi import Ui_Form
import os
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from modules.PandasModel import PandasModel
from modules.asdreader import reader
import matplotlib.pyplot as plt

POSTFIX = '_ViewData'
from os import path
from modules import Utils

class ViewDataASD(Ui_Form):

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
        super(ViewDataASD, self).setupUi(Form)
        self.Form = Form
        self.radioButton_6.setVisible(False)
        self.model = None
        self.connectWidgets()



    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveas())
        self.pushButton_4.clicked.connect(lambda: self.saveTableData(self.data))
        self.plotBtn.clicked.connect(lambda: self.plotData())

        self.radioButton.toggled.connect(lambda: self.refreshTable())
        self.radioButton_2.toggled.connect(lambda: self.refreshTable())
        self.radioButton_3.toggled.connect(lambda: self.refreshTable())
        self.radioButton_6.toggled.connect(lambda: self.refreshTable())

    def refreshTable(self):
        if self.model is not None:
            # self.tableView.removeRows()
            self.tableView.model().removeAllDataFrameRows()

    def get_measureType(self):
        measure_type=None
        if (self.radioButton.isChecked() == True):
            measure_type="raw"
        elif (self.radioButton_2.isChecked() == True):
            measure_type = "reflectance"
        elif (self.radioButton_3.isChecked() == True):
            measure_type = "radiance"
        elif (self.radioButton_6.isChecked() == True):
            measure_type = "white_reference"
        return measure_type

    def loadDataASD(self):

        measure_type=self.get_measureType()
        self.populateTable(measure_type)

    def browseButton_clicked(self):
        self.lineEdit_2.setText("")
        self.listWidget.clear()

        fnames = Utils.browseMultipleFiles(POSTFIX + '.csv', "Supported types (*.asd)", self.lineEdit_2, self.lineEdit)
        if fnames:
            for filename in fnames:
                self.listWidget.addItem(os.path.basename(filename))
            self.refreshTable()

    def populateTable(self,measure_type):
        # try:
        try:
            df = pd.DataFrame()
            df2 = pd.DataFrame()
            df2.index.name = 'wavelength'
            filepaths=self.lineEdit_2.text().split(";")
            filenames = []
            for file in filepaths:
                df1 = df.append(self.readASDclass(file, measure_type))
                df1.index.name = 'wavelength'
                df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)
                filenames.append(os.path.basename(file))

            df2.columns = filenames
            df2 = df2.reset_index()
            self.data = df2
            self.model = PandasModel(df2.T)
            self.tableView.setModel(self.model)
        except:
            QtWidgets.QMessageBox.information(self.Form, 'Message', 'Data may not not contain '+measure_type, QtWidgets.QMessageBox.Ok)





    def readASDclass(self, filename, function):
        df = reader(filename)
        data = pd.DataFrame(df.__getattr__(function))
        data.index = df.wavelengths
        return data

    def plotData(self):
        if self.model is None:
            QtWidgets.QMessageBox.information(self.Form, 'Message', 'Please load data first.', QtWidgets.QMessageBox.Ok)
            return

        if self.tableView.model().rowCount()<=0:
            QtWidgets.QMessageBox.information(self.Form, 'Message', 'Please load data first.',
                                              QtWidgets.QMessageBox.Ok)
            return

        filepaths=self.lineEdit_2.text().split(';')
        df2=self.data
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(0, len(filepaths)):
            label = os.path.basename(filepaths[i])
            ax.plot(df2.iloc[:, 0], df2.iloc[:, i + 1], label=label)
            ax.legend()

        plt.xlabel("wavelength")
        measure_type=self.get_measureType()
        plt.ylabel(measure_type)
        plt.show()



    def saveTableData(self, df):

        model = self.tableView.model()
        if model is None:
            messageDisplay = "Empty data cannot be saved!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (model.rowCount()==0):
            messageDisplay = "Empty data cannot be saved!"
            QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        df.to_csv(self.outputFilename, index=False)
        messageDisplay = "Data saved Sucessfully in the output file!"
        QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)

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
            QtWidgets.QMessageBox.information(self.Form, 'Error', "Kindly enter a valid output path.", QtWidgets.QMessageBox.Ok)
            return

        if not str(self.lineEdit.text()).split(".")[-1]=='csv':
            self.lineEdit.setFocus()
            messageDisplay = "Output file extension cannot be " + str(self.lineEdit_2.text()).split(".")[-1]
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return


        filepath = str(self.lineEdit_2.text()).split(";")


        for file in filepath:
            if (not os.path.exists(file)):
                messageDisplay = "Path does not exist : "+file
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if not str(os.path.basename(file).split('.')[-1]) == "asd":
                messageDisplay = "Input file extension cannot be " + str(os.path.basename().split('.')[-1])
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        self.inFile = self.lineEdit_2.text()
        self.outputFilename = self.lineEdit.text()

        print("In: " + self.inFile)
        print("Out: " + self.outputFilename)
        print("Running...")


        self.loadDataASD()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    ui = ViewData()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
