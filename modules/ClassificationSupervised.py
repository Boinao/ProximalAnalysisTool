# -*- coding: utf-8 -*-
"""
***************************************************************************
    ClassificationSupervised.py
    ---------------------
    Date                 : November 2020
    Author               : Anand SS, Ross, Nidhin
    Email                : anandss@isro.gov.in
***************************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from Ui.ClassificationUi import Ui_Form

import os
import sys
import pandas as pd
from scipy import stats
from PyQt5 import QtWidgets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture as gmm

from sklearn import linear_model
from os import path

from modules import Utils

POSTFIX = '_Classification'
class ClassificationSupervised(Ui_Form):
    '''
       This module perform Supervised Classification
    '''
    
    def __init__(self):
        self.curdir=None
        self.filepath=[]
        self.output1=pd.DataFrame()
        self.outputFilename=""

    
    def get_widget(self):
        return self.groupBox
    
    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()
    
    def setupUi(self, Form):
        super(ClassificationSupervised,self).setupUi(Form)
        self.Form = Form
        self.connectWidgets()
        
    
    
    def connectWidgets(self):
        self.label_3.show()
        self.lineEdit.show()
        self.lineEdit_2.show()
        self.label.show()
        self.label_3.setText("Gamma      ")
        self.label.setText("C Value    ")
        self.groupBox_2.hide()
        self.comboBox_2.hide()
        self.comboBox_3.hide()
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.comboBox.currentIndexChanged.connect(lambda: self.changeVisibility( ))

        self.browseMetaBtn.clicked.connect(lambda: self.browseMetadata())

    def changeVisibility(self):
        if(self.comboBox.currentText()=="SVM RBF" or self.comboBox.currentText()=="SVM Linear" or self.comboBox.currentText()=='SVM Poly'):
            self.label_3.show()
            self.lineEdit.show()
            self.lineEdit_2.show()
            self.label.show()
            self.label_3.setText("Gamma      ")
            self.label.setText("C Value    ")
            self.lineEdit_2.setText("1")
            self.lineEdit.setText("0.01")
            self.groupBox_2.hide()
            self.comboBox_2.hide()
            self.comboBox_3.hide()
        if(self.comboBox.currentText()=="Gausian Mixture Model"):
            self.label_3.show()
            self.lineEdit.show()
            self.lineEdit_2.hide()
            self.label.hide()
            self.label_3.setText("Components")
            self.lineEdit.setText("1")
            self.groupBox_2.show()
            self.comboBox_2.show()
            self.comboBox_2.clear()
            self.comboBox_2.addItem("Select Covariance Type")
            self.comboBox_2.addItem("full")
            self.comboBox_2.addItem("tied")
            self.comboBox_2.addItem("diag")
            self.comboBox_2.addItem("spherical")
            self.comboBox_3.hide()
        if(self.comboBox.currentText()=="KNN"):
            self.label_3.show()
            self.lineEdit.show()
            self.lineEdit_2.hide()
            self.label.hide()
            self.label_3.setText("Neighbors: ")
            self.lineEdit.setText("5")
            self.groupBox_2.hide()
            self.comboBox_2.hide()
            self.comboBox_3.hide()
        if(self.comboBox.currentText()=="Multinomial Logistic"):
            self.label_3.hide()
            self.lineEdit.hide()
            self.lineEdit_2.hide()
            self.label.hide()
            self.groupBox_2.show()
            self.comboBox_2.show()
            self.comboBox_2.clear()
            self.comboBox_2.addItem("Select Solver Options")
            self.comboBox_2.addItem("newton-cg")
            self.comboBox_2.addItem("lbfgs")
            self.comboBox_2.addItem("sag")
            self.comboBox_3.show()
            self.comboBox_3.clear()
            self.comboBox_3.addItem("Select Multiclass Options")
            self.comboBox_3.addItem("ovr")
            self.comboBox_3.addItem("multinomial")
        if(self.comboBox.currentText()=="Random Forest"):
            self.label_3.show()
            self.lineEdit.show()
            self.lineEdit_2.show()
            self.label.show()
            self.label_3.setText("Max Depth  ")
            self.label.setText("Estimators ")
            self.lineEdit_2.setText("10")
            self.lineEdit.setText("6")
            self.groupBox_2.hide()
            self.comboBox_2.hide()
            self.comboBox_3.hide()

    def browseMetadata(self):
        Utils.browseMetadataFile(self.metadataTxt,"Supported types (*.csv)")
        #
        # fname = []
        # lastDataDir = Utils.getLastUsedDir()
        #
        # self.metadataTxt.setText("")
        # fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)
        #
        # if not fname:
        #     return
        # self.metafilepath = fname
        # if fname:
        #     self.metadataTxt.setText(fname)
        #     Utils.setLastUsedDir(os.path.dirname(fname))
        # else:
        #     self.metadataTxt.setText("")

    def browseButton_clicked(self):
        Utils.browseInputFile(POSTFIX +".txt",self.lineEdit_3,"Supported types (*.csv)",self.lineEdit_4)

        # fname = []
        # lastDataDir = Utils.getLastUsedDir()
        #
        # self.lineEdit_3.setText("")
        # fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)
        #
        # if not fname:
        #     return
        #
        # self.filepath = fname
        #
        # # print(self.filepath)
        # if fname:
        #     self.lineEdit_3.setText(fname)
        #     Utils.setLastUsedDir(os.path.dirname(fname))
        #
        #     self.outputFilename = (os.path.dirname(fname)) + "/Output" + POSTFIX +".txt"
        #     self.lineEdit_4.setText(self.outputFilename)
        # else:
        #     self.lineEdit_3.setText("")

    def saveasButton_clicked(self):
        Utils.browseSaveFile(self.lineEdit_4,'*.txt')
        # lastDataDir = Utils.getLastSavedDir()
        # self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.txt')
        # if self.outputFilename:
        #     self.lineEdit_4.setText(self.outputFilename)
        #
        # Utils.setLastSavedDir(os.path.dirname(self.outputFilename))
        #
        # return self.outputFilename

    

    def SVM(self, kernelName,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=self.testSize,random_state=101)
        gamma=float(self.lineEdit.text())

        clf=SVC(kernel=kernelName,C=float(self.lineEdit_2.text()),gamma=gamma)
        clf.fit(x_train,y_train)
        Score=clf.score(x_test,y_test)
        Score=str(Score)
        yfit = clf.predict(x_test)  
        
        Creport=self.confusionMatrix(y_test,yfit)
        finalOutput=str("Classification Report:\n"+Creport+"\nAccuracy Score: \n"+str(Score))

        with open(self.outputFilename,"w") as f:
            f.write(finalOutput)
        print("Classification report written to :", self.outputFilename)

    def RandomForest(self,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=self.testSize,random_state=101)
        max_depth=int(self.lineEdit.text())
        noOfEstimators=int(self.lineEdit_2.text())
        model = RandomForestClassifier(max_depth=max_depth,n_estimators=noOfEstimators, random_state=0)
        model.fit(x_train,y_train) 
        model.score(x_train,y_train)
        Score=model.score(x_test,y_test)
        Score=str(Score)
        yfit = model.predict(x_test)  
        Creport=self.confusionMatrix(y_test,yfit)
        finalOutput=str("Classification Report:\n"+Creport+"\nAccuracy Score: \n"+str(Score))
        with open(self.outputFilename,"w") as f:
            f.write(finalOutput)
        print("Classification report written to :", self.outputFilename)
            

    def MultinomialLogisticRegression(self,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=self.testSize,random_state=101)

        solver_options=str(self.comboBox_2.currentText())
        multi_class_options=str(self.comboBox_3.currentText())
        lr=linear_model.LogisticRegression(solver=solver_options,multi_class=multi_class_options,)
        lr.fit(x_train,y_train)
    
        Score=lr.score(x_test,y_test)
        Score=str(Score)
        yfit = lr.predict(x_test)  
        Creport=self.confusionMatrix(y_test,yfit)
        finalOutput=str("Classification Report:\n"+Creport+"\nAccuracy Score: \n"+str(Score))
        with open(self.outputFilename,"w") as f:
            f.write(finalOutput)
        print("Classification report written to :", self.outputFilename)
    
    def KNN(self,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=self.testSize,random_state=101,stratify=y)
        neighbor=int(self.lineEdit.text())
        if(len(x_train)<neighbor):
                    messageDisplay="No. of neighbors ("+str(neighbor)+") should be less than equal to No. of Samples ("+str(len(x_train))+")."
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
        clf=KNeighborsClassifier(n_neighbors=neighbor)
        clf.fit(x_train,y_train)
        clf.score(x_test,y_test)
        Score=clf.score(x_test,y_test)
        Score=str(Score)
        yfit = clf.predict(x_test)  
        Creport=self.confusionMatrix(y_test,yfit)
        finalOutput=str("Classification Report:\n"+Creport+"\nAccuracy Score: \n"+str(Score))
        self.outputFilename = self.lineEdit_3.text()
        with open(self.outputFilename,"w") as f:
            f.write(finalOutput)
        print("Classification report written to :", self.outputFilename)
    
    def Gaussian(self,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=self.testSize,random_state=101)
        noOfComponents=int(self.lineEdit.text())
        if(len(x_train)<noOfComponents):
                    messageDisplay="No. of components ("+str(noOfComponents)+") should be less than equal to No. of Samples ("+str(len(x_train))+")."
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
        covType=str(self.comboBox_2.currentText())
        
        gm=(gmm(n_components=int(noOfComponents),covariance_type=covType,random_state=0).fit(x_train))
        ypred=gm.predict(x_test)
        accuracy=np.mean(ypred.ravel()==y_test.ravel())
        Score=str(accuracy)
        Creport=self.confusionMatrix(y_test,ypred)
        finalOutput=str("Classification Report:\n"+Creport+"\nAccuracy Score: \n"+str(Score))
        with open(self.outputFilename,"w") as f:
            f.write(finalOutput)

        print("Classification report written to :", self.outputFilename)
            
    def confusionMatrix(self,ytest,yfit):
        Creport=str(classification_report(ytest, yfit))
        mat = confusion_matrix(ytest, yfit)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()
        return Creport
        
        

    
    
    def run(self):

        #Validate input field
        if not Utils.validataInputPath(self.lineEdit_3, self.Form):
            return

        # if (self.lineEdit_3.text() is None) or (self.lineEdit_3.text() == ""):
        #     self.lineEdit_3.setFocus()
        #     QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
        #                                   QtWidgets.QMessageBox.Ok)
        #     return
        #
        # if (not os.path.exists(self.lineEdit_3.text())):
        #     self.lineEdit_3.setFocus()
        #     QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid", "Kindly enter a valid input file.",
        #                                    QtWidgets.QMessageBox.Ok)
        #     return

        #Validate Metadata
        if not Utils.validataInputPath(self.metadataTxt, self.Form):
            return

        # if (self.metadataTxt.text() is None) or (self.metadataTxt.text() == ""):
        #     self.metadataTxt.setFocus()
        #     QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
        #                                   QtWidgets.QMessageBox.Ok)
        #     return
        #
        # if (not os.path.exists(self.metadataTxt.text())):
        #     self.metadataTxt.setFocus()
        #     QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid", "Kindly enter a valid file.",
        #                                    QtWidgets.QMessageBox.Ok)
        #     return

        # Validate Output
        if not Utils.validataOutputPath(self.lineEdit_4, self.Form):
            return

        # if (self.lineEdit_4.text() is None) or (self.lineEdit_4.text() == ""):
        #     self.lineEdit_4.setFocus()
        #     QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Output File is required",
        #                                   QtWidgets.QMessageBox.Ok)
        #     return
        #
        # if (not os.path.isdir(os.path.dirname(self.lineEdit_4.text()))):
        #     self.lineEdit_4.setFocus()
        #     QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
        #                                    "Kindly enter a valid output path.",
        #                                    QtWidgets.QMessageBox.Ok)
        #     return

        # Validate Empty
        if not Utils.validateEmpty(self.lineEdit_5, self.Form):
            return

        # if(self.lineEdit_5.text()=="" ):
        #     self.lineEdit_5.setFocus()
        #     messageDisplay="Test Size cannot be left empty!"
        #     QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
        #     return
        try:
            float(self.lineEdit_5.text())
        except Exception:
            self.lineEdit_5.setFocus()
            self.lineEdit_5.selectAll()
            messageDisplay="Only Float or Integer values allowed"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        if(float(self.lineEdit_5.text())>0 and float(self.lineEdit_5.text())<1):
            self.testSize=float(self.lineEdit_5.text())
        else:
            self.lineEdit_5.setFocus()
            messageDisplay="Enter value in the range 0 to 1 only!"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return

        filepath = self.lineEdit_3.text()
        metafilepath = self.metadataTxt.text()
        self.outputFilename = self.lineEdit_4.text()

        print("In: " + filepath)
        print("Out: " +self.outputFilename)
        print("Meta data: " + metafilepath)
        print("Running...")


        df_spectra = pd.read_csv(filepath, header=0, index_col=0)
        df_metadata = pd.read_csv(metafilepath, header=None, index_col=0)

        data = df_spectra.to_numpy().T
        y = df_metadata.loc['class_label'].values.astype(np.int8)

        if (self.checkBox.isChecked() == True):
            pca = PCA()
            principle_components = pca.fit_transform(data)
            data = pd.DataFrame(principle_components)
            x = data
        else:
            x = data.copy()
        if (self.comboBox.currentText() == "SVM RBF" or self.comboBox.currentText() == "SVM Linear" or self.comboBox.currentText() == "SVM Poly"):
            if (self.lineEdit.text() == ""):
                self.lineEdit.setFocus()
                messageDisplay = "Gamma value cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.lineEdit_2.text() == ""):
                self.lineEdit_2.setFocus()
                messageDisplay = "C value cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            try:
                int(self.lineEdit_2.text())
            except Exception:
                self.lineEdit_2.setFocus()
                messageDisplay = "C can only be of Integer type!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            try:
                float(self.lineEdit.text())
            except Exception:
                self.lineEdit.setFocus()
                messageDisplay = "Gamma can only be of Float or Integer type!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (int(self.lineEdit_2.text()) <= 0):
                self.lineEdit_2.setFocus()
                self.lineEdit_2.selectAll()
                messageDisplay = "Enter positive values only!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (float(self.lineEdit.text()) <= 0):
                self.lineEdit.setFocus()
                self.lineEdit.selectAll()
                messageDisplay = "Enter positive values only!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return

        if (self.comboBox.currentText() == "SVM RBF"):
            self.SVM("rbf", x, y)
        if (self.comboBox.currentText() == "SVM Linear"):
            self.SVM("linear", x, y)
        if (self.comboBox.currentText() == "SVM Poly"):
            self.SVM("poly", x, y)
        if (self.comboBox.currentText() == "Random Forest"):
            if (self.lineEdit.text() == ""):
                messageDisplay = "Max Depth cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.lineEdit_2.text() == ""):
                messageDisplay = "Estimators cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            try:
                int(self.lineEdit_2.text())
            except Exception:
                messageDisplay = "Estimators can only be of Integer type!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            try:
                int(self.lineEdit.text())
            except Exception:
                messageDisplay = "Max Depth can only be of Integer type!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (int(self.lineEdit_2.text()) < 0 or float(self.lineEdit.text()) < 0):
                messageDisplay = "Enter positive values only!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            self.RandomForest(x, y)

        if (self.comboBox.currentText() == "Multinomial Logistic"):
            if (self.comboBox_2.currentText() == "Select Solver Options"):
                messageDisplay = "Select appropriate solver option!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.comboBox_3.currentText() == "Select Multiclass Options"):
                messageDisplay = "Select appropriate multiclass option!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            self.MultinomialLogisticRegression(x, y)

        if (self.comboBox.currentText() == "KNN"):
            if (self.lineEdit.text() == ""):
                messageDisplay = "Neighbors cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            try:
                int(self.lineEdit.text())
            except Exception:
                messageDisplay = "Neighbors can only be of Integer type!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (int(self.lineEdit.text()) < 0):
                messageDisplay = "Enter positive values only!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            self.KNN(x, y)

        if (self.comboBox.currentText() == "Gausian Mixture Model"):
            if (self.comboBox_2.currentText() == "Select Covariance Type"):
                messageDisplay = "Select appropriate covariance type!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (self.lineEdit.text() == ""):
                messageDisplay = "Components cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            try:
                int(self.lineEdit.text())
            except Exception:
                messageDisplay = "Components can only be of Integer type!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            if (int(self.lineEdit.text()) < 0):
                messageDisplay = "Enter positive values only!"
                QtWidgets.QMessageBox.information(self.Form, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
                return
            self.Gaussian(x, y)



