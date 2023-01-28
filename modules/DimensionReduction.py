# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:10:20 2019

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
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.DimensionalityReductionUi import Ui_Form
import os
import sys
import pandas as pd
from scipy import stats
from PyQt5 import QtWidgets
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
import seaborn as sns 
from sklearn.manifold import TSNE
from os import path



POSTFIX = '_Dimension'
from modules import Utils
class DimensionReduction(Ui_Form):
    
    def __init__(self):
        self.curdir=None
        self.filepath=[]
        self.output1=pd.DataFrame()
        self.outputFilename=""
        self.output2=pd.DataFrame()
        self.outputFilename2=""

    
    def get_widget(self):
        return self.groupBox
    
    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()
    
    def setupUi(self, Form):
        super(DimensionReduction, self).setupUi(Form)
        self.Form = Form

        # super(DimensionReduction,self).setupUi(Form)
        # self.Form = Form
        self.connectWidgets()
        self.widget_feature_6.hide()


    def default_visibility(self):
        self.checkBox.hide()
        self.label_6.hide()
        self.label_10.hide()
        self.label_8.hide()
        self.label_9.hide()
        self.lineEdit_7.hide()
        self.lineEdit_6.hide()
        self.lineEdit_5.hide()
        self.groupBox_2.hide()
        self.radioButton.hide()
        self.radioButton_2.hide()
        self.radioButton_3.hide()

    
    
    def connectWidgets(self):
        self.default_visibility()

        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.run())
        self.browseMetaBtn.clicked.connect(lambda: self.browseMetadata())
        self.comboBox.currentIndexChanged.connect(lambda: self.changeVisibility( ))

   
    def changeVisibility(self):
               
        if(self.comboBox.currentText()=="PCA"):
            self.default_visibility()

        elif(self.comboBox.currentText()=="Kernel PCA"):
            self.default_visibility()
            self.groupBox_2.show()
            self.radioButton.show()
            self.radioButton_2.show()
            self.radioButton_3.show()

            self.radioButton.setText("linear")
            self.radioButton_2.setText("rbf")
            self.radioButton_3.setText("poly")

#            self.radioButton.setText()
        elif(self.comboBox.currentText()=="LLE"):
            self.default_visibility()
            self.checkBox.show()
            self.label_6.show()
            self.label_8.show()
            self.lineEdit_5.show()
            self.groupBox_2.show()
            self.groupBox_2.setTitle("Eigen Solver:")
            self.radioButton.show()
            self.radioButton_2.show()
            self.radioButton_3.show()
            self.radioButton.setText("auto")
            self.radioButton_2.setText("arpack")
            self.radioButton_3.setText("dense")
        elif(self.comboBox.currentText()=="ISOMAP"):
            self.default_visibility()
            self.label_8.show()
            self.lineEdit_5.show()
            self.groupBox_2.show()
            self.groupBox_2.setTitle("Eigen Solver:")
            self.radioButton.show()
            self.radioButton_2.show()
            self.radioButton_3.show()

            self.radioButton.setText("auto")
            self.radioButton_2.setText("arpack")
            self.radioButton_3.setText("dense")
        elif(self.comboBox.currentText()=="t-SNE"):
            self.default_visibility()
            self.label_10.show()
            self.label_9.show()
            self.lineEdit_7.show()
            self.lineEdit_6.show()

        elif(self.comboBox.currentText()=="Select Method"):
            self.default_visibility()

            
    def browseMetadata(self):
        Utils.browseMetadataFile(self.metadataTxt, "Supported types (*.csv)")

    def browseButton_clicked(self):
        Utils.browseInputFile(POSTFIX + ".csv", self.lineEdit_3, "Supported types (*.csv)", self.lineEdit_4)

    def get_data(self):
        filepath = self.lineEdit_3.text()
        metafilepath = self.metadataTxt.text()
        df_spectra = pd.read_csv(filepath, header=0, index_col=0)
        df_metadata = pd.read_csv(metafilepath, header=None, index_col=0)

        spectra = df_spectra.to_numpy().T
        classes = df_metadata.loc['class_label'].values.astype(np.int8)
        # class_name = list(set(df_metadata.loc['class'].values))
        class_name = df_metadata.loc['class'].values

        return spectra,classes, class_name

    def PCA(self):
        self.widget_feature_6.ax.clear()
        n_comp = int(self.lineEdit.text())

        spectra, classes, class_name=self.get_data()
        pca = PCA(n_components=n_comp)
        principle_components = pca.fit_transform(spectra)
        exp = pca.explained_variance_ratio_
        comp = pca.components_
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        # self.widget_feature_6.ax.scatter(principle_components[:, 0], principle_components[:, 1], c=classes, s=50)
        # self.widget_feature_6.ax.set_xlabel("Feature 0")
        # self.widget_feature_6.ax.set_ylabel("Feature 1")
        # self.widget_feature_6.canvas.draw()
        self.saveToFile(principle_components, classes)
        self.showPlot(principle_components, class_name)


        # result = pd.DataFrame(principle_components)
        # result['y']=classes
        # result.to_csv(self.outputFilename2)

        # plt.figure()
        # plt.scatter(principle_components[:, 0], principle_components[:, 1], c=classes, s=50)
        # plt.xlabel('Feature 0')
        # plt.ylabel('Feature 1')
        # plt.show()

        # QtWidgets.QMessageBox.information(self.Form, 'Information', "Results saved to output file", QtWidgets.QMessageBox.Ok)

    def showPlot(self, X_transformed,class_name):
        f, ax=plt.subplots(1)
        sns.scatterplot(X_transformed[:, 0], X_transformed[:, 1], ax=ax,
                        hue=class_name)
        plt.xlabel('Component 0')
        plt.ylabel('Component 1')
        plt.show()

        plt.show()
        #
        # plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=classes, s=50)

    def saveToFile(self,X_transformed, classes):
        result = pd.DataFrame(X_transformed)
        result['y'] = classes
        result.to_csv(self.outputFilename2)


    def KernelPCA(self):
        self.widget_feature_6.ax.clear()

        kernelName = "linear"
        if (self.radioButton.isChecked() == True):
            kernelName = "linear"
        elif (self.radioButton_2.isChecked() == True):
            kernelName = "rbf"
        elif (self.radioButton_3.isChecked() == True):
            kernelName = "poly"

        n_comp = int(self.lineEdit.text())

        spectra, classes, class_name=self.get_data()

        transformer = KernelPCA(kernel=kernelName, n_components=n_comp)
        X_transformed = transformer.fit_transform(spectra)
        self.saveToFile(X_transformed,classes)
        self.showPlot(X_transformed, class_name)

    def LLE(self):
        self.widget_feature_6.ax.clear()
        n_comp = int(self.lineEdit.text())
        neighbor = int(self.lineEdit_5.text())

        if (neighbor < 1):
            messageDisplay = "Neighbors cannot be less than 1!!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        eigenSolver = "auto"
        if (self.radioButton.isChecked() == True):
            eigenSolver = "auto"
        elif (self.radioButton_2.isChecked() == True):
            eigenSolver = "arpack"
        elif (self.radioButton_3.isChecked() == True):
            eigenSolver = "dense"
        randomState = 45
        if (self.checkBox.isChecked() == False):
            randomState = np.random

        spectra, classes, class_name=self.get_data()

        embedding = LocallyLinearEmbedding(random_state=randomState, n_neighbors=neighbor, n_components=n_comp,
                                           eigen_solver=eigenSolver)
        X_transformed = embedding.fit_transform(spectra)

        self.saveToFile(X_transformed, classes)
        self.showPlot(X_transformed, class_name)

        # QtWidgets.QMessageBox.information(self.Form, 'Information', "Results saved to output file",
        #                                   QtWidgets.QMessageBox.Ok)



        
    def ISOMAP(self):
        self.widget_feature_6.ax.clear()
        n_comp = int(self.lineEdit.text())
        if (n_comp < 2):
            messageDisplay = "Components cannot be less than 2!!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        neighbor = int(self.lineEdit_5.text())
        if (neighbor < 1):
            messageDisplay = "Neighbors cannot be less than 1!!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return
        eigenSolver = "auto"
        if (self.radioButton.isChecked() == True):
            eigenSolver = "auto"
        elif (self.radioButton_2.isChecked() == True):
            eigenSolver = "arpack"
        elif (self.radioButton_3.isChecked() == True):
            eigenSolver = "dense"

        spectra, classes, class_name=self.get_data()

        embedding = Isomap(n_neighbors=neighbor, n_components=n_comp, eigen_solver=eigenSolver)
        X_transformed = embedding.fit_transform(spectra)

        # self.widget_feature_6.ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=classes, s=50)
        # self.widget_feature_6.ax.set_xlabel("Feature 0")
        # self.widget_feature_6.ax.set_ylabel("Feature 1")
        # self.widget_feature_6.canvas.draw()

        self.saveToFile(X_transformed, classes)
        self.showPlot(X_transformed, class_name)
        # QtWidgets.QMessageBox.information(self.Form, 'Information', "Results saved to output file",
        #                                   QtWidgets.QMessageBox.Ok)


        
    def TSNE(self):
        self.widget_feature_6.ax.clear()
        n_comp = int(self.lineEdit.text())


        spectra, classes, class_name=self.get_data()

        X_transformed = TSNE(n_components=n_comp, perplexity=Perplexity, learning_rate=learningRate).fit_transform(X)

        self.saveToFile(X_transformed, classes)
        self.showPlot(X_transformed, class_name)




        
    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename,_=QFileDialog.getSaveFileName(None,'save',lastDataDir,'*.csv')
        if not self.outputFilename:
            return

        self.lineEdit_4.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename

    def checkNeighbours(self):
        return (Utils.validateEmpty(self.lineEdit_5, self.Form) and \
                Utils.validateDatatype(self.lineEdit_5, int, self.Form))
    
    def run(self):

        if not (Utils.validataInputPath(self.lineEdit_3, self.Form) and \
                Utils.validataOutputPath(self.lineEdit_4, self.Form) and \
                Utils.validateExtension(self.lineEdit_3, 'csv', self.Form) and \
                Utils.validateExtension(self.lineEdit_4, 'csv', self.Form) and \
                Utils.validateCombobox(self.comboBox,self.Form) and \
                Utils.validateEmpty(self.lineEdit, self.Form) and \
                Utils.validateDatatype(self.lineEdit, int, self.Form) and \
                Utils.validateRange(self.lineEdit, 2)
                ):
            return

        self.outputFilename2=self.lineEdit_4.text()

        if(self.comboBox.currentText()=="PCA"):
            self.PCA()
        if(self.comboBox.currentText()=="Kernel PCA"):
            self.KernelPCA()
        if(self.comboBox.currentText()=="LLE"):
            if not self.checkNeighbours():
                return
            self.LLE()
            
        if(self.comboBox.currentText()=="ISOMAP"):
            if not self.checkNeighbours():
                return
            self.ISOMAP()
            
        if(self.comboBox.currentText()=="t-SNE"):
            if not (Utils.validateEmpty(self.lineEdit_6, self.Form) and \
                    Utils.validateEmpty(self.lineEdit_7, self.Form) and \
                    Utils.validateDatatype(self.lineEdit_6, int, self.Form) and \
                    Utils.validateDatatype(self.lineEdit_7, int, self.Form) and \
                    Utils.validateRange(self.lineEdit_6, 5, 50) and \
                    Utils.validateRange(self.lineEdit_7, 100, 1000)):
                return
            self.TSNE()

        
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()()
    #QSizePolicy sretain=Form.sizePolicy()
    #sretain.setRetainSizeWhenHidden(True)
    #sretain.setSizePolicy()
    ui = DimensionReduction()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
