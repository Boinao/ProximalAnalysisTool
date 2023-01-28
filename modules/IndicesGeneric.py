# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:05:06 2019

@author: Trainee
"""

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5 import QtWidgets
from Ui.indices import Ui_Form
import os
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from . import indices

from os import path

POSTFIX = '_Generic'
from modules import Utils

pluginPath = os.path.split(os.path.dirname(__file__))[0]


class IndicesGeneric(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.vegindex = 0.0
        self.filename = ""

    def get_widget(self):
        return self.groupBox

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def setupUi(self, Form):
        path_index=os.path.join(pluginPath, 'external', 'Vegetation_index.csv')
        ind = pd.read_csv(path_index)
        super(IndicesGeneric, self).setupUi(Form)
        self.Form = Form
        for i in range(0, 56):
            item = ind.iloc[i, 0]
            self.listWidget.addItem(item)

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveas())
        self.groupBox.setTitle("Generic Spectral Indices")

    def browseButton_clicked(self):
        # Utils.browseInputFile(POSTFIX + ".csv", self.lineEditInput, "Supported types (*.csv)", self.lineEditOutput)
        fnames = Utils.browseMultipleFiles(POSTFIX + '.csv', "Supported types (*.csv)", self.lineEditInput,
                                           self.lineEditOutput)
        if not fnames:
            self.lineEditInput.setText("")
            self.lineEditOutput.setText("")

    def saveas(self):
        self.filename = Utils.browseSaveFile(self.lineEditOutput, '*.csv')

    def run(self):

        Utils.run_spectral_indices('Generic',self.lineEditInput, self.lineEditOutput, self.listWidget, self.Form)

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    ui = SpectralIndices()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
