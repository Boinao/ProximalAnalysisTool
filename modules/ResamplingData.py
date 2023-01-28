# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:47:06 2019

@author: Trainee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:20:53 2019

@author: Trainee
"""

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
import seaborn as sns
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.ResamplingUi import Ui_Form
import os
import sys
import pandas as pd
from PyQt5 import QtWidgets
from matplotlib import pyplot as plt
from modules.asdreader import reader
from spectral import resampling

from os import path
POSTFIX = '_Resample'
class Resample(Ui_Form):
    
    def __init__(self):
        self.curdir=None
        self.filepath=""
        self.output1=pd.DataFrame()
        self.outputFilename=""
        self.filepath2=""
        self.filepath3=""
    
    def get_widget(self):
        return self.groupBox
    
    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()
    
    def setupUi(self, Form):
        super(Resample,self).setupUi(Form)
        self.Form = Form
        
        self.connectWidgets()
        
    
    
    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.pushButton_3.clicked.connect(lambda: self.centralWavelength_clicked())
        self.pushButton_4.clicked.connect(lambda: self.fwhm_clicked())

    
    def browseButton_clicked(self):
        
        if self.curdir is None:
            self.curdir = os.getcwd()
            self.curdir=self.curdir.replace("\\","/")
            
        self.lineEdit_3.setText("")
        fname,_=QFileDialog.getOpenFileName(None,'Open File',self.curdir,'*.asd')
        if(len(fname)==0):
            messageDisplay="Select atleast 1 file!"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.filepath=fname
        
        #print(self.filepath)
        if fname:
                inputText=str(fname)
                self.lineEdit_3.setText(inputText)
                    
                self.outputFilename=(str(self.curdir))+"/Output"+POSTFIX+".csv"
                self.lineEdit_4.setText(self.outputFilename)
        else:
                self.lineEdit_3.setText("")
                
    def centralWavelength_clicked(self):
        fname=""
        if self.curdir is None:
            self.curdir = os.getcwd()
            
        self.lineEdit.setText("")
        fname,_=QFileDialog.getOpenFileName(None,'Open File',self.curdir,'*.txt')
        if(fname==""):
            messageDisplay="Select atleast 1 file!"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.filepath2=fname
        
        #print(self.filepath)
        if fname:
                inputText=str(fname)
                self.lineEdit.setText(inputText)
                    
        else:
                self.lineEdit.setText("")
                
    def fwhm_clicked(self):
        
        if self.curdir is None:
            self.curdir = os.getcwd()
            
        self.lineEdit_2.setText("")
        fname,_=QFileDialog.getOpenFileName(None,'Open File',self.curdir,'*.txt')
        if(fname==""):
            messageDisplay="Select atleast 1 file!"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        
        self.filepath3=fname
        
        #print(self.filepath)
        if fname:
                inputText=str(fname)
                self.lineEdit_2.setText(inputText)

        else:
                self.lineEdit_2.setText("")
   
    def resampler(self):
        df=reader(self.filepath)
        data=pd.DataFrame(df.__getattr__("reflectance"))
        data.index=df.wavelengths
        val1=pd.read_table(self.filepath3, delim_whitespace=True,   index_col=0)
        fwhm=np.float64(val1.iloc[:,0])
        val2=pd.read_table(self.filepath2, delim_whitespace=True,  index_col=0)
        centralWavelength=np.float64(val2.iloc[:,0])
        output=resampling.BandResampler(df.wavelengths,centralWavelength,fwhm2=fwhm)
        resampled=output.__call__(data)
#        plt.plot(df.wavelengths,data)
#        plt.plot(centralWavelength,resampled)
        resampledDF=pd.DataFrame(resampled)
        finalOutput=pd.DataFrame(columns=['Resampled values'])
        finalOutput.loc[:,'Resampled values']=resampledDF.iloc[:,0]
        finalOutput.index=centralWavelength
        print(finalOutput)
        finalOutput.to_csv(self.outputFilename)
        fig, ax = plt.subplots(figsize=(10, 5))
        label=(self.filepath.split('/')[-1].split('.')[0])
#        plt.plot(df.wavelengths,data,label=label, centralWavelength,resampled,label="Resampled")
#        plt.plot(df.wavelengths,data,'--', centralWavelength,resampled,'-')
        ax.plot(df.wavelengths,data,label=label)
        ax.plot(centralWavelength,resampled,label="Resampled")
        ax.legend()
        plt.xticks(df.wavelengths[::150])
        plt.xlabel("Wavelength")
        plt.ylabel("reflectance")
#        xtick=list(df2.index)
#        ax.set_xticks(xtick[::100]) 
#        plt.xlabel("wavelength")
#        if(operation=='raw'):
#             plt.ylabel('DN')   
#        else:
#            plt.ylabel(operation)   
        
    def saveasButton_clicked(self):
        self.outputFilename,_=QFileDialog.getSaveFileName(None,'save',self.curdir,'*.png')
        if self.outputFilename:
            self.lineEdit_4.setText(self.outputFilename)
   
        return self.outputFilename
    
    
    def run(self):
        if(self.lineEdit_3.text()==""):
                messageDisplay="Cannot leave Input empty!"
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
        filepath=str(self.lineEdit_3.text())
        print(filepath)
        if(str(filepath.split('/')[-1].split('.')[-1])=="asd"):
                pass
        else:
                self.lineEdit_3.setFocus()
                self.lineEdit_3.selectAll()
                messageDisplay="Input file extension cannot be "+str(filepath.split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return 
        if(path.exists(filepath.rsplit('/',1)[0])):
            pass
        
        else:
            messageDisplay="ASD File Path does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        if(path.isfile(filepath)):
            pass
        else:
            messageDisplay="ASD File does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.filepath=filepath
        """Check central wavelength file"""
        filepath=str(self.lineEdit.text())
        if(path.exists(filepath.rsplit('/',1)[0])):
            pass
        else:
            messageDisplay="File Path for central wavelength does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        if(path.isfile(filepath)):
            pass
        else:
            messageDisplay="File for central wavelength does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.filepath2=filepath
        filepath=str(self.lineEdit_2.text())
        if(path.exists(filepath.rsplit('/',1)[0])):
            pass
        else:
            messageDisplay="File Path for fwhm does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        if(path.isfile(filepath)):
            pass
        else:
            messageDisplay="File for fwhm does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.filepath3=filepath
        if(self.lineEdit_4.text()==""):
                self.lineEdit_4.setFocus()
                self.lineEdit_4.selectAll()
                messageDisplay="Cannot leave Output empty!"
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
            
        outputPath=str(self.lineEdit_4.text()).split()
        if(len(outputPath)==1):
            pass
        else:
             messageDisplay="Enter one ouput file only!"
             QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
             return
        if(path.exists(outputPath[0].rsplit('/',1)[0])):
            if(str(outputPath[0].split('/')[-1].split('.')[-1])=="csv"):
                pass
            else:
                self.lineEdit_4.setFocus()
                self.lineEdit_4.selectAll()
                messageDisplay="Output file extension cannot be "+str(outputPath[0].split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
        else:
            self.lineEdit_4.setFocus()
            self.lineEdit_4.selectAll()
            messageDisplay="Output File Path does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.outputFilename=outputPath[0]
        
        self.resampler()
       

        
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    #QSizePolicy sretain=Form.sizePolicy()
    #sretain.setRetainSizeWhenHidden(True)
    #sretain.setSizePolicy()
    ui = Resample()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
