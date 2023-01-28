# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:56:35 2019

@author: Trainee
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:57:57 2019

@author: Trainee
"""

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import seaborn as sns
# mutual_info_classif, mutual_info_regression: Functions for calculating Mutual Information Between classes and the target
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from Ui.SpectralDistancesUi import Ui_Form
import os
import sys
import pandas as pd
from scipy import stats
from PyQt5 import QtWidgets
from sklearn.decomposition import PCA
from os import path
from modules import Utils

POSTFIX = '_Dist'


class spectralDistance(Ui_Form):

    def __init__(self):
        self.curdir = None
        # self.filepath=""
        self.filepath = []
        self.output1 = pd.DataFrame()
        self.threshold = 0
        self.outputFilename = ""

    def get_widget(self):
        return self.groupBox

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def setupUi(self, Form):
        super(spectralDistance, self).setupUi(Form)
        self.Form = Form
        self.connectWidgets()

    def connectWidgets(self):

        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.browseMetaBtn.clicked.connect(lambda: self.browseMetadata())

    def browseMetadata(self):
        Utils.browseMetadataFile(self.medataTxt, "Supported types (*.csv)")

    def browseButton_clicked(self):
        Utils.browseInputFile(POSTFIX + ".csv", self.lineEdit_3, "Supported types (*.csv)", self.lineEdit_4)

    def getData(self):

        filepath = self.lineEdit_3.text()
        metafilepath = self.medataTxt.text()
        df_spectra = pd.read_csv(filepath, header=0, index_col=0)
        df_metadata = pd.read_csv(metafilepath, header=None, index_col=0)

        spectra = df_spectra.to_numpy().T
        classes = df_metadata.loc['class_label'].values.astype(np.int8)
        labels = df_metadata.loc['class'].values
        classes_unique = np.unique(classes)
        return spectra, classes, labels, classes_unique

    def calculateDistance(self, distance, outputFile):
        from scipy.spatial.distance import correlation
        fraction = 2.0
        spectra, classes, labels, classes_unique = self.getData()
        d = np.zeros((len(classes_unique), len(classes_unique)))
        for i in range(1, len(classes_unique) + 1):
            for j in range(1, len(classes_unique) + 1):
                X = spectra[np.where(classes == i)]
                Y = spectra[np.where(classes == j)]

                if distance == 'JM Distance':
                    B = -np.log(np.nansum([np.sqrt((p) * (q)) for (p, q) in zip(X, Y)]))
                    d[i - 1, j - 1] = np.sqrt(2.0 * (1.0 - np.exp(B)))
                elif distance == 'Manhattan':
                    d[i - 1, j - 1] = np.sum([np.fabs(p - q) for (p, q) in zip(X, Y)])
                elif distance == 'Euclidean':
                    d[i - 1, j - 1] = np.linalg.norm([(p - q) for (p, q) in zip(X, Y)])
                elif distance == 'Cosine':
                    d[i - 1, j - 1] = 1.0-np.mean(
                        ([np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)) for (p, q) in zip(X, Y)]))
                elif distance == 'Correlation':
                    d[i - 1, j - 1] = correlation(X.mean(axis=0), Y.mean(
                        axis=0))  # 1-np.mean([((p-np.mean(p))-(q-np.mean(q))) for (p,q) in zip(X,Y)])/np.sqrt(np.mean([np.square((p-np.mean(p)))*np.square((q-np.mean(q))) for (p,q) in zip(X,Y) ]))
                elif distance == 'Chebyshev':
                    d[i - 1, j - 1] = np.amax(np.amax(np.abs(X - Y)))
                elif distance == 'SID':
                    d[i - 1, j - 1] = np.nansum([((a / np.sum(a)) + np.spacing(1)) * np.log(
                        (((a / np.sum(a)) + np.spacing(1))) / ((b / np.sum(b)) + np.spacing(1))) + (
                                                         (b / np.sum(b)) + np.spacing(1)) * np.log(
                        (((b / np.sum(b)) + np.spacing(1))) / ((a / np.sum(a)) + np.spacing(1))) for (a, b) in
                                                 zip(X, Y)])
                elif distance == 'Fractional':
                    d[i - 1, j - 1] = np.amax(
                        np.power(np.sum([(p - q) ** fraction for (p, q) in zip(X, Y)]), (1 / fraction)))
                # print("Distance between", np.unique(labels[np.where(classes == i)][0]), "and ",
                #       np.unique(labels[np.where(classes == j)][0]), " ")
                # print(d[i - 1, j - 1])

        headers = [np.unique(labels[np.where(classes == i)])[0] for i in range(1, len(classes_unique) + 1)]
        result = pd.DataFrame(d, columns=headers)
        print(result)
        result.to_csv(outputFile)

    def saveasButton_clicked(self):
        Utils.browseSaveFile(self.lineEdit_4, '*.csv')

    def run(self):

        if not (Utils.validataInputPath(self.lineEdit_3, self.Form) and \
                Utils.validataInputPath(self.medataTxt, self.Form) and \
                Utils.validataOutputPath(self.lineEdit_4, self.Form) and \
                Utils.validateCombobox(self.comboBox, self.Form)):
            return

        inFile = self.lineEdit_3.text()
        outputFile = self.lineEdit_4.text()
        metaFile = self.medataTxt.text()

        print("In: " + inFile)
        print("Metadata: " + metaFile)
        print("Out: " + outputFile)
        print("Running...")

        self.calculateDistance(self.comboBox.currentText(), outputFile)
        print("Successfully completed module.")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    # QSizePolicy sretain=Form.sizePolicy()
    # sretain.setRetainSizeWhenHidden(True)
    # sretain.setSizePolicy()
    ui = spectralDistance()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
