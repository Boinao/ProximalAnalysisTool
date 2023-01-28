# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:33:38 2019

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
from sklearn.model_selection import train_test_split
# mutual_info_classif, mutual_info_regression: Functions for calculating Mutual Information Between classes and the target
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.featureSelectionMultipleClassesUi import Ui_Form
import os
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from scipy import stats
from PyQt5 import QtWidgets
from os import path
from modules import Utils
POSTFIX = '_Feature'
class FeatureSelectionMultipleClasses(Ui_Form):
    
    def __init__(self):
        self.curdir=None
        #self.filepath=""
        self.filepath=[]
        self.Targetfilepath=""
        self.output1=pd.DataFrame()
        self.threshold=0
        self.outputFilename=""

    
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
        super(FeatureSelectionMultipleClasses,self).setupUi(Form)
        self.Form = Form
        self.connectWidgets()
        
    
    
    def connectWidgets(self):
        self.pushButton_3.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton.clicked.connect(lambda: self.browseTargetClass())

        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.comboBox.currentIndexChanged.connect(lambda: self.changeVisibility( ))
        #if(self.comboBox.currentText()=="SDI"):
           #self.label_7.hide()
     
        #self.pushButton_3.clicked.connect(lambda: self.saveasKButton_clicked())
    def changeVisibility(self):
        if(self.comboBox.currentText()=="ANOVA"):
            self.label_6.show()
            self.lineEdit_2.show()
        if(self.comboBox.currentText()=="Kruskal"):
            self.label_6.show()
            self.lineEdit_2.show()
        if(self.comboBox.currentText()=="SDI"):
            self.label_6.hide()
            self.lineEdit_2.hide()
            self.lineEdit.setText("1")
        
    def browseTargetClass(self):
        fname=""
        # if self.curdir is None:
        #     self.curdir = os.getcwd()
        #     self.curdir=self.curdir.replace("\\","/")
        lastDataDir = Utils.getLastUsedDir()
        self.lineEdit_3.setText("")
        fname,_=QFileDialog.getOpenFileName(None,'Open Target File',lastDataDir,"*.csv")
        if(len(fname)==0):
            # messageDisplay="Select 1 Target Class!"
            # QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.Targetfilepath=fname
        
        #print(self.filepath)
        
        if fname:
                self.lineEdit_3.setText(str(fname))
                Utils.setLastUsedDir(os.path.dirname(fname))
                self.outputFilename = os.path.dirname(fname) +"/Output"+POSTFIX+".csv"
                self.lineEdit_5.setText(self.outputFilename)
        else:
                self.lineEdit_3.setText("")
        
        
    def browseButton_clicked(self):
        #print("HI")
        fname=[]
        if self.curdir is None:
            self.curdir = os.getcwd()
            
        self.lineEdit_4.setText("")
        fname,_=QFileDialog.getOpenFileNames(None,filter="Supported types (*.csv)",directory=self.curdir)
        if(len(fname)==0):
            messageDisplay="Select atleast 1 class!"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.filepath=fname
        
        #print(self.filepath)
        filenames=""
        if fname:
                for i in fname:
                    filenames=filenames+str(i)+" "
                self.lineEdit_4.setText(str(filenames))
                self.outputFilename=(str(self.curdir))+"\\Order"+POSTFIX+".csv"
                self.lineEdit_5.setText(self.outputFilename)
        else:
                self.lineEdit_3.setText("")
        
   
        
        

        
    def ANOVA(self):
        self.threshold=self.lineEdit.text()
        df1=pd.read_csv(self.Targetfilepath)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df1.select_dtypes(include=numerics).columns)
        data1 = df1[numerical_features]
        data1=data1.transpose()
        alpha=float(self.lineEdit_2.text())
        print(alpha)
        SumOfOutputs=pd.DataFrame(columns=['K Best'])
        for i in range(len(data1)):
            SumOfOutputs.loc[i,'K Best']=0
        SumOfOutputs.index=data1.index
        for i in range (0,len(self.filepath)):
            df2=pd.read_csv(self.filepath[i])
            numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
            numerical_features = list(df2.select_dtypes(include=numerics).columns)
            data2 = df2[numerical_features]
            data2=data2.transpose()
            out=pd.DataFrame(columns=['F val','P val','K Best','BEST'])
            for j in range(len(data1)):
                f,p=stats.f_oneway(data1.iloc[j,:],data2.iloc[j,:])
                out.loc[j,'F val']=f
                out.loc[j,'P val']=p
            out.index=data1.index    
            out['K Best']=0
            output=out.sort_values('F val',ascending=False,inplace=False)
#            print(output)
            count=0
            for i in range(len(out)):
                if(output.iloc[i,1]<alpha and count<int(self.threshold)):
                    output.iloc[i,2]=1
                    count=count+1
            output.index=output.index.astype(float)
            output.sort_index(inplace=True)
            SumOfOutputs.iloc[:,0] = SumOfOutputs.iloc[:,0] + output.iloc[:,2]
    
        SumOfOutputs=SumOfOutputs/len(self.filepath)
        SumOfOutputs.sort_values('K Best',ascending=False,inplace=True)
        counter=0
        for c in range(len(SumOfOutputs)):
            if(SumOfOutputs.iloc[c,0]==1):
                SumOfOutputs.iloc[c,0]=1
                counter=counter+1
            else:
                SumOfOutputs.iloc[c,0]=0
                
        if(counter<int(self.threshold)):
            messageDisplay="Only "+str(counter)+" values selected from total of "+str(self.threshold)+" values."
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
        SumOfOutputs.index=SumOfOutputs.index.astype(float)
        SumOfOutputs.sort_index(inplace=True)
        self.output1=SumOfOutputs
        print(SumOfOutputs)
        self.output1.to_csv(self.outputFilename)
        df1=pd.read_csv(self.Targetfilepath)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df1.select_dtypes(include=numerics).columns)
        data = df1[numerical_features]
        data=data.transpose()
        ax2=data.iloc[:,0].plot()
        for i in self.filepath:
            df1=pd.read_csv(i)
            numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
            numerical_features = list(df1.select_dtypes(include=numerics).columns)
            data = df1[numerical_features]
            data=data.transpose()
            ax2=data.iloc[:,0].plot()
        ax2=SumOfOutputs['K Best'].plot(kind='bar')
        ticks=(ax2.xaxis.get_ticklocs())
        ticklabels=[n.get_text() for n in ax2.xaxis.get_ticklabels()]
        for i in range(0,len(SumOfOutputs)):
            ticklabels[i]=float(("%.2f")%float(ticklabels[i]))
        if(i>1000):
            ax2.xaxis.set_ticks(ticks[::100])
            ax2.xaxis.set_ticklabels(ticklabels[::100])
        elif(i<1000 and i>500):
            ax2.xaxis.set_ticks(ticks[::50])
            ax2.xaxis.set_ticklabels(ticklabels[::50])
        else:
            ax2.xaxis.set_ticks(ticks[::10])
            ax2.xaxis.set_ticklabels(ticklabels[::10])
#        ax2.xaxis.set_ticks(ticks[::10])
#        ax2.xaxis.set_ticklabels(ticklabels[::10])  
        ax2.set_autoscaley_on(False)
        ax2.set_ylim([0,len(self.filepath)+1]) 
        plt.xlabel("Wavelength")
        plt.ylabel("F Value")
        
        
        
    def kruskal(self):
        self.threshold=self.lineEdit.text()
        df1=pd.read_csv(self.Targetfilepath)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df1.select_dtypes(include=numerics).columns)
        data1 = df1[numerical_features]
        data1=data1.transpose()
        alpha=float(self.lineEdit_2.text())
        print(alpha)
        SumOfOutputs=pd.DataFrame(columns=['K Best'])
        for i in range(len(data1)):
            SumOfOutputs.loc[i,'K Best']=0
        SumOfOutputs.index=data1.index
        for i in range (0,len(self.filepath)):
            df2=pd.read_csv(self.filepath[i])
            numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
            numerical_features = list(df2.select_dtypes(include=numerics).columns)
            data2 = df2[numerical_features]
            data2=data2.transpose()
            out=pd.DataFrame(columns=['F val','P val','K Best','BEST'])
            for j in range(len(data1)):
                f,p=stats.kruskal(data1.iloc[j,:],data2.iloc[j,:])
                out.loc[j,'F val']=f
                out.loc[j,'P val']=p
            out.index=data1.index    
            out['K Best']=0
            output=out.sort_values('F val',ascending=False,inplace=False)
            count=0
            for i in range(len(out)):
                if(output.iloc[i,1]<alpha and count<int(self.threshold)):
                    output.iloc[i,2]=1
                    count=count+1
            output.index=output.index.astype(float)
            output.sort_index(inplace=True)
            SumOfOutputs.iloc[:,0] = SumOfOutputs.iloc[:,0] + output.iloc[:,2]
    
        SumOfOutputs=SumOfOutputs/len(self.filepath)
        SumOfOutputs.sort_values('K Best',ascending=False,inplace=True)
        counter=0
        for c in range(len(SumOfOutputs)):
            if(SumOfOutputs.iloc[c,0]==1):
                SumOfOutputs.iloc[c,0]=1
                counter=counter+1
            else:
                SumOfOutputs.iloc[c,0]=0
                
        if(counter<int(self.threshold)):
            messageDisplay="Only "+str(counter)+" values selected from total of "+str(self.threshold)+" values."
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
        SumOfOutputs.index=SumOfOutputs.index.astype(float)
        SumOfOutputs.sort_index(inplace=True)
        self.output1=SumOfOutputs
        print(SumOfOutputs)
        self.output1.to_csv(self.outputFilename)
        df1=pd.read_csv(self.Targetfilepath)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df1.select_dtypes(include=numerics).columns)
        data = df1[numerical_features]
        data=data.transpose()
        ax2=data.iloc[:,0].plot()
        for i in self.filepath:
            df1=pd.read_csv(i)
            numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
            numerical_features = list(df1.select_dtypes(include=numerics).columns)
            data = df1[numerical_features]
            data=data.transpose()
            ax2=data.iloc[:,0].plot()
        ax2=SumOfOutputs['K Best'].plot(kind='bar')
        ticks=(ax2.xaxis.get_ticklocs())
        ticklabels=[n.get_text() for n in ax2.xaxis.get_ticklabels()]
        for i in range(0,len(SumOfOutputs)):
            ticklabels[i]=float(("%.2f")%float(ticklabels[i]))
        if(i>1000):
            ax2.xaxis.set_ticks(ticks[::100])
            ax2.xaxis.set_ticklabels(ticklabels[::100])
        elif(i<1000 and i>500):
            ax2.xaxis.set_ticks(ticks[::50])
            ax2.xaxis.set_ticklabels(ticklabels[::50])
        else:
            ax2.xaxis.set_ticks(ticks[::10])
            ax2.xaxis.set_ticklabels(ticklabels[::10])
#        ax2.xaxis.set_ticks(ticks[::10])
#        ax2.xaxis.set_ticklabels(ticklabels[::10])  
        ax2.set_autoscaley_on(False)
        ax2.set_ylim([0,len(self.filepath)+1]) 
        plt.xlabel("Wavelength")
        plt.ylabel("F Value")
        
    def SDI(self):
        df1=pd.read_csv(self.Targetfilepath)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df1.select_dtypes(include=numerics).columns)
        data1 = df1[numerical_features]
        data1=data1.transpose()
        ax2=data1.iloc[:,0].plot()
        threshold=float(self.lineEdit.text())
        print(threshold)
        SumOfOutputs=pd.DataFrame(columns=['Best'])
        for i in range(len(data1)):
            SumOfOutputs.loc[i,'Best']=0
        SumOfOutputs.index=data1.index
        for i in range (0,len(self.filepath)):
            df2=pd.read_csv(self.filepath[i])
            numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
            numerical_features = list(df2.select_dtypes(include=numerics).columns)
            data2 = df2[numerical_features]
            data2=data2.transpose()
            sdi=(abs(df1.mean()-df2.mean()))/(df1.std()+df2.std())        
            sdi=pd.DataFrame(sdi)   
            sdi['Best']=0
            output=sdi.sort_values(sdi.columns[0],ascending=False,inplace=False)
            count=0
            for j in range(len(output)):
                if(output.iloc[j,0]>threshold):
                    output.iloc[j,1]=1
                    count=count+1
            print(count)
            output.index=output.index.astype(float)
            output.sort_index(inplace=True)
            SumOfOutputs.iloc[:,0] = SumOfOutputs.iloc[:,0] + output.iloc[:,1]
            
        SumOfOutputs=SumOfOutputs/(len(self.filepath))
        SumOfOutputs.sort_values('Best',ascending=False,inplace=True)
        for c in range(len(SumOfOutputs)):
            if(SumOfOutputs.iloc[c,0]==1):
                SumOfOutputs.iloc[c,0]=1
            else:
                SumOfOutputs.iloc[c,0]=0
        SumOfOutputs.index=SumOfOutputs.index.astype(float)
        SumOfOutputs.sort_index(inplace=True)
#        filepaths=[]
#        for i in range(0,len(self.filepath)):
#            filepaths.append( )
        for i in self.filepath:
            df1=pd.read_csv(i)
            numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
            numerical_features = list(df1.select_dtypes(include=numerics).columns)
            data = df1[numerical_features]
            data=data.transpose()
            ax2=data.iloc[:,0].plot()
            #label=str(i.split('/')[-1].split('.')[0])
        
        ax2=SumOfOutputs['Best'].plot(kind='bar',color=np.where(SumOfOutputs['Best']==1,'r','g'))
        ticks=(ax2.xaxis.get_ticklocs())
        ticklabels=[n.get_text() for n in ax2.xaxis.get_ticklabels()]
        for i in range(0,len(SumOfOutputs)):
            ticklabels[i]=float(("%.2f")%float(ticklabels[i]))
        if(i>1000):
            ax2.xaxis.set_ticks(ticks[::100])
            ax2.xaxis.set_ticklabels(ticklabels[::100])
        elif(i<1000 and i>500):
            ax2.xaxis.set_ticks(ticks[::50])
            ax2.xaxis.set_ticklabels(ticklabels[::50])
        else:
            ax2.xaxis.set_ticks(ticks[::10])
            ax2.xaxis.set_ticklabels(ticklabels[::10])
#        ax2.xaxis.set_ticks(ticks[::10])
#        ax2.xaxis.set_ticklabels(ticklabels[::10])  
        ax2.set_autoscaley_on(False)
        ax2.set_ylim([0,len(self.filepath)]) 
#        ax2.legend()
        plt.xlabel("Wavelength")
        plt.ylabel("SDI")

      
    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename,_=QFileDialog.getSaveFileName(None,'save',lastDataDir,'*.csv')

        if not self.outputFilename:
            return


        if self.outputFilename:
            #self.output1.to_csv(self.outputFilename)
            self.lineEdit_5.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename
    
    
    def run(self):
        if(self.lineEdit_3.text()==""):
                self.lineEdit_3.setFocus()
                self.lineEdit_3.selectAll()
                messageDisplay="Cannot leave Input empty!"
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
       
        if(self.lineEdit_4.text()==""):
                self.lineEdit_4.setFocus()
                self.lineEdit_4.selectAll()
                messageDisplay="Cannot leave Input empty!"
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
            
        targetfilepath=str(self.lineEdit_3.text())
                
        if(str(targetfilepath.split('/')[-1].split('.')[-1])=="csv"):
                pass
        else:
                self.lineEdit_3.setFocus()
                self.lineEdit_3.selectAll()
                messageDisplay="Input file extension cannot be "+str(targetfilepath.split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return 
        if(path.exists(targetfilepath.rsplit('/',1)[0])):
            pass
        else:
            self.lineEdit_3.setFocus()
            self.lineEdit_3.selectAll()
            messageDisplay="Target File Path does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        if(path.isfile(targetfilepath)):
            pass
        else:
            messageDisplay="Target File does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        filepath=str(self.lineEdit_4.text()).split()
        for i in filepath:         
            if(str(i.split('/')[-1].split('.')[-1])=="csv"):
                pass
            else:
                self.lineEdit_4.setFocus()
                self.lineEdit_4.selectAll()
                messageDisplay="Input file extension cannot be "+str(i.split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return 
#        print(filepath)
        c=0
        count=0
        print(len(filepath))
        for i in range(len(filepath)):
                if(len(self.filepath)==0):
                    if (path.exists(filepath[i].rsplit('/',1)[0])==True):
                        pass
                    else:
                        messageDisplay="Path does not exist!"
                        QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                        return
                    if (path.isfile(filepath[i])==True):
                        count=count+1
                        pass
                    else:
                        messageDisplay="File does not Exist!"
                        QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                        return
                else:
                    if(filepath[i]==self.filepath[i]):
                        continue
                    else:
                        c=1
                        break
       
        if(count==i+1):
            self.filepath=filepath
        if(c==1):
                self.lineEdit_4.setFocus()
                self.lineEdit_4.selectAll()
                for i in filepath:
#                    print(i.rsplit('/',1)[0])
#                    print(i)
                    my_path=path.exists(i.rsplit('/',1)[0])
                    if (path.exists(i.rsplit('/',1)[0])==True):
                        pass
                    else:
                        messageDisplay="Path does not exist!"
                        QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                        return
                    my_file = path.isfile(i)
                    if (path.isfile(i)==True):
                        pass
                    else:
                        messageDisplay="File does not Exist!"
                        QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                        return
                self.filepath=filepath
#        else:
#            print(self.filepath)
            
        if(self.lineEdit_5.text()==""):
                self.lineEdit_5.setFocus()
                self.lineEdit_5.selectAll()
                messageDisplay="Cannot leave Output empty!"
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
            
        outputPath=str(self.lineEdit_5.text()).split()
        if(len(outputPath)==1):
            pass
        else:
             messageDisplay="Enter one ouput file only!"
             QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
             return
        if(path.exists(outputPath[0].rsplit('/',1)[0])):
            print(outputPath[0].split('/')[-1].split('.')[1])
            if(str(outputPath[0].split('/')[-1].split('.')[-1])=="csv"):
                pass
            else:
                self.lineEdit.setFocus()
                self.lineEdit.selectAll()
                messageDisplay="Output file extension cannot be "+str(outputPath[0].split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
        else:
            self.lineEdit_5.setFocus()
            self.lineEdit_5.selectAll()
            messageDisplay="Output File Path does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.outputFilename=outputPath[0]
      
        if(self.lineEdit.text()==""):
            messageDisplay="Threshold value cannot be left empty!"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        
        if(self.comboBox.currentText()=="ANOVA"):
            if(self.lineEdit_2.text()==""):
                messageDisplay="Alpha value cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
            try:
                    int(self.lineEdit.text())
            except Exception:
                    messageDisplay="Threshold can only be of Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
            try:
                    float(self.lineEdit_2.text())
            except Exception:
                    messageDisplay="Alpha can only be of Float or Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
            self.ANOVA()
            
        if(self.comboBox.currentText()=="Kruskal"):
            if(self.lineEdit_2.text()==""):
                messageDisplay="Alpha value cannot be left empty!"
                QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
            try:
                    int(self.lineEdit.text())
            except Exception:
                    messageDisplay="Threshold can only be of Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
            try:
                    float(self.lineEdit_2.text())
            except Exception:
                    messageDisplay="Alpha can only be of Float or Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
            self.kruskal()
            
        if(self.comboBox.currentText()=="SDI"):
            try:
                    float(self.lineEdit.text())
            except Exception:
                    messageDisplay="Threshold can only be of Float or Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
            self.SDI()
            self.label_6.hide()
            
            
                

        
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    ui = FeatureSelectionMultipleClasses()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
