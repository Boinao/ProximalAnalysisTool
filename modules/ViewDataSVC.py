# -*- coding: utf-8 -*-
"""
Created on Fri december 11 14:05:06 2020

@author: Nidhin
"""
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import specdal
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

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

class ViewDataSVC(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.outputFilename = ""
        self.data = pd.DataFrame()
        self.model = None

    def get_widget(self):
        return self.groupBox

    def delete(self):
        pass

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def setupUi(self, Form):
        super(ViewDataSVC, self).setupUi(Form)
        self.Form = Form
        self.svcWhiteRb.setVisible(False)
        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveas())
        self.svcSaveBtn.clicked.connect(lambda: self.saveTableSVC(self.data))
        self.svcPlotBtn.clicked.connect(lambda: self.plotDataSVC())

        # self.radioButton.toggled.connect(lambda: self.reloadData())
        self.svcReflRb.toggled.connect(lambda: self.refreshTable())
        self.svcRadRb.toggled.connect(lambda: self.refreshTable())
        self.svcWhiteRb.toggled.connect(lambda: self.refreshTable())

    def refreshTable(self):
        if self.model is not None:
            self.tableView.model().removeAllDataFrameRows()

    def loadDataSVC(self):
        files=self.lineEdit_2.text().split(";")
        self.data, self.model=Utils.call_loadData(files,self.svcReflRb,self.svcRadRb,self.svcWhiteRb,self.tableView, spec='SVC')

    def plotDataSVC(self):
        if self.model is None:
            QtWidgets.QMessageBox.information(self.Form, 'Message', 'Please load data first.', QtWidgets.QMessageBox.Ok)
            return

        if self.tableView.model().rowCount()<=0:
            QtWidgets.QMessageBox.information(self.Form, 'Message', 'Please load data first.',
                                              QtWidgets.QMessageBox.Ok)
            return

        files = self.lineEdit_2.text().split(";")
        Utils.call_plotData(files,self.svcReflRb,self.svcRadRb,self.svcWhiteRb,spec='SVC')

    def saveTableSVC(self, df):

        Utils.saveTable(df, self.tableView, self.lineEdit.text())

    def browseButton_clicked(self):
        self.lineEdit_2.setText("")
        self.listWidget.clear()


        fnames = Utils.browseMultipleFiles(POSTFIX + '.csv',"Supported types (*.sig)", self.lineEdit_2, self.lineEdit)
        if fnames:
            for filename in fnames:
                self.listWidget.addItem(os.path.basename(filename))
            self.refreshTable()


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


        files = self.lineEdit_2.text().split(";")
        print(files)
        for file in files:
            if (not os.path.exists(file)):
                messageDisplay = "Path does not exist : " + file
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

            if not str(os.path.basename(file).split('.')[-1]) == "sig":
                messageDisplay = "Input file extension cannot be " + str(os.path.basename().split('.')[-1])
                QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        # self.outputFilename = str(self.lineEdit.text())
        # self.selectOperation()

        self.inFile = self.lineEdit_2.text()
        self.outputFilename = self.lineEdit.text()

        print("In: " + self.inFile)
        print("Out: " + self.outputFilename)
        print("Running...")

        self.loadDataSVC()
        # pass


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    ui = SigNewreader()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
