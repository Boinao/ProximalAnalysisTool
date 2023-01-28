# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:05:06 2019

@author: Trainee
"""
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from Ui.indices import Ui_Form
import os
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from . import indices

from os import path
pluginPath = os.path.split(os.path.dirname(__file__))[0]
POSTFIX = '_ASD'

from modules import Utils
class IndicesASD(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.vegindex = 0.0
        self.filename = ""

    def get_widget(self):
        return self.groupBox

    def delete(self):
        """
        Empty code in case of removal of the widget what needs to done
        """
        pass

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def setupUi(self, Form):
        path = os.path.join(pluginPath, 'external', 'Vegetation_index_sig_asd.csv')
        ind = pd.read_csv(path)
        super(IndicesASD, self).setupUi(Form)
        self.Form = Form
        for i in range(0, 55):
            item = ind.iloc[i, 0]
            self.listWidget.addItem(item)

        self.connectWidgets()

    def connectWidgets(self):
        self.groupBox.setTitle("ASD Spectral Indices")
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveas())

    def browseButton_clicked(self):
        fnames = Utils.browseMultipleFiles(POSTFIX + '.csv', "Supported types (*.asd)", self.lineEditInput, self.lineEditOutput)
        if not fnames:
            self.lineEditInput.setText("")
            self.lineEditOutput.setText("")
        # Utils.browseInputFile(POSTFIX + ".csv", self.lineEditInput, "Supported types (*.asd)", self.lineEditOutput)

    def saveas(self):
        self.filename=Utils.browseSaveFile(self.lineEditOutput,'*.csv')

    def run(self):

        Utils.run_spectral_indices('ASD',self.lineEditInput,self.lineEditOutput,self.listWidget,self.Form)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    ui = qtIndices()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
