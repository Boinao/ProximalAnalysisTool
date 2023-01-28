# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:57:57 2019

@author: Trainee
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
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

from Ui.FeatureSelectionUI import Ui_Form
import os
import sys
import pandas as pd
from scipy import stats
from PyQt5 import QtWidgets
from os import path
from modules import Utils
POSTFIX = '_Feature'
class FeatureSelection(Ui_Form):
    
    def __init__(self):
        self.curdir=None
        #self.filepath=""
        self.filepath=[]
        self.output1=pd.DataFrame()
        self.k_Best=pd.DataFrame()
        self.threshold=0
        self.outputFilename=""
        self.outputKfilename=""

    
    def get_widget(self):
        return self.groupBox

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()
    
    def setupUi(self, Form):
        super(FeatureSelection,self).setupUi(Form)
        self.Form = Form
        self.connectWidgets()
        
    
    
    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.comboBox.currentIndexChanged.connect(lambda: self.changeVisibility( ))

    def changeVisibility(self):
#        if(self.comboBox.currentText()=="JM Distance"):
#            self.label_6.hide()
#            self.lineEdit.hide()
#            self.lineEdit_2.hide()
#            self.label.hide()
        if(self.comboBox.currentText()=="Normalised Index Selection"):
            self.label_6.hide()
            self.lineEdit.hide()
            self.lineEdit_2.hide()
            self.label.hide()
        if(self.comboBox.currentText()=="Mutual Information"):
            self.label_6.hide()
            self.lineEdit.show()
            self.lineEdit_2.hide()
            self.label.show()
        if(self.comboBox.currentText()=="F Classification"):
            self.label_6.hide()
            self.lineEdit.show()
            self.lineEdit_2.hide()
            self.label.show()
        if(self.comboBox.currentText()=="Kruskal"):
            self.label_6.show()
            self.lineEdit.show()
            self.lineEdit_2.show()
            self.label.show()
        if(self.comboBox.currentText()=="ANOVA"):
            self.label_6.show()
            self.lineEdit.show()
            self.lineEdit_2.show()
            self.label.show()
        if(self.comboBox.currentText()=="SDI"):
            self.label_6.hide()
            self.lineEdit.show()
            self.lineEdit_2.hide()
            self.label.show()
        
    def browseButton_clicked(self):
        fname=[]
        # if self.curdir is None:
        #     self.curdir = os.getcwd()
        #     self.curdir=self.curdir.replace("\\","/")
        lastDataDir = Utils.getLastUsedDir()
        self.lineEdit_3.setText("")
        fname,_=QFileDialog.getOpenFileNames(None,filter="Supported types (*.csv)",directory=lastDataDir)
        if(len(fname)>2 or len(fname)<2):
            messageDisplay="Select 2 Classes!"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.filepath=fname
        
        if fname:
                inputText=str(fname[0])+" "
                for i in range(1,len(fname)):
                    inputText=inputText+" "+fname[i]
                self.lineEdit_3.setText(inputText)
                Utils.setLastUsedDir(os.path.dirname(fname[0]))
                self.outputFilename = os.path.dirname(fname[0]) +"/Output"+POSTFIX+".csv"
                self.lineEdit_4.setText(self.outputFilename)
        else:
                self.lineEdit_3.setText("")
        
        
        
    def mutualInformation(self):
        self.threshold=self.lineEdit.text()
        #print(int(self.threshold))
        df1=pd.read_csv(self.filepath[0])
        df2=pd.read_csv(self.filepath[1])
        df1['Y']=0
        df2['Y']=1
        l3=df1.append(df2,ignore_index=True)
        df=pd.DataFrame(l3)
        #df = pd.read_csv(self.filepath)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df.select_dtypes(include=numerics).columns)
        data = df[numerical_features]
        X = data.drop(['Y'], axis=1)
        y = data['Y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        #X_train.shape, y_train.shape, X_test.shape, y_test.shape
        self.XTrain=X_train
        self.yTrain=y_train
        mutual_info = mutual_info_classif(X_train.fillna(0), y_train)
        mi_series = pd.Series(mutual_info)
        mi_series.index = X_train.columns
        mi_series.sort_values(ascending=False)
        #mi.sort_values(ascending=False).plot.bar(figsize=(20,8))
        uni=pd.DataFrame(mi_series.sort_values(ascending=False, inplace=False))
        #uni=uni.drop(['Unnamed: 0'])
        uni['Best']=0
        val=int(self.threshold)
        for i in range(0,val):
            uni.iloc[i,1]=1
        self.output1=uni
        self.output1.to_csv(self.outputFilename)
        import numpy as np
        uni.index=uni.index.astype(float)
        uni=uni.sort_index()
        ax=uni[uni.columns[0]].plot(kind='bar',color=np.where(uni['Best']==1,'g','r'))   
        ticks=ax.xaxis.get_ticklocs()
        ticklabels=[n.get_text() for n in ax.xaxis.get_ticklabels()]
        for i in range(0,len(uni)):
            ticklabels[i]=float(("%.2f")%float(ticklabels[i]))
        if(i>1000):
            ax.xaxis.set_ticks(ticks[::100])
            ax.xaxis.set_ticklabels(ticklabels[::100])
        elif(i<1000 and i>500):
            ax.xaxis.set_ticks(ticks[::50])
            ax.xaxis.set_ticklabels(ticklabels[::50])
        else:
            ax.xaxis.set_ticks(ticks[::10])
            ax.xaxis.set_ticklabels(ticklabels[::10])
#        ax.xaxis.set_ticks(ticks[::10])
#        ax.xaxis.set_ticklabels(ticklabels[::10])    
        plt.xlabel("Wavelength")    
        plt.ylabel("Mutual Information")  
        #ax=miDF[miDF.columns[0]].plot(kind="bar",title=" hey ",figsize=(20,8),legend=True)
        #plt.show()
        k_best_features = SelectKBest(mutual_info_classif, k=int(self.threshold)).fit(X_train.fillna(0), y_train)
        self.k_Best=pd.DataFrame(X_train.columns[k_best_features.get_support()])
        #print(self.k_Best)
        
    def fclassification(self):
        self.threshold=self.lineEdit.text()
        df1=pd.read_csv(self.filepath[0])
        df2=pd.read_csv(self.filepath[1])
        df1['Y']=0
        df2['Y']=1
        l3=df1.append(df2,ignore_index=True)
        df=pd.DataFrame(l3)
        #df = pd.read_csv(self.filepath)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df.select_dtypes(include=numerics).columns)
        data = df[numerical_features]
        X = data.drop(['Y'], axis=1)

        y = data['Y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        X_train.shape, y_train.shape, X_test.shape, y_test.shape

        univariate = f_classif(X_train.fillna(0), y_train)
        univariate = pd.Series(univariate[1])
        univariate.index = X_train.columns
        univariate.sort_values(ascending=False, inplace=True)
        
        uni=pd.DataFrame(univariate.sort_values(ascending=False, inplace=False))
        #uni=uni.drop(['Unnamed: 0'])
        uni['Best']=0
        val=int(self.threshold)
        for i in range(0,val):
                uni.iloc[i,1]=1
        import numpy as np
        uni.index=uni.index.astype(float)
        uni=uni.sort_index()
        self.output1=uni
        self.output1.to_csv(self.outputFilename)
        ax=uni[uni.columns[0]].plot(kind='bar',color=np.where(uni['Best']==1,'g','r'))   
        ticks=ax.xaxis.get_ticklocs()
        ticklabels=[n.get_text() for n in ax.xaxis.get_ticklabels()]
        for i in range(0,len(uni)):
            ticklabels[i]=float(("%.2f")%float(ticklabels[i]))
        if(i>1000):
            ax.xaxis.set_ticks(ticks[::100])
            ax.xaxis.set_ticklabels(ticklabels[::100])
        elif(i<1000 and i>500):
            ax.xaxis.set_ticks(ticks[::50])
            ax.xaxis.set_ticklabels(ticklabels[::50])
        else:
            ax.xaxis.set_ticks(ticks[::10])
            ax.xaxis.set_ticklabels(ticklabels[::10])
#        ax.xaxis.set_ticks(ticks[::10])
#        ax.xaxis.set_ticklabels(ticklabels[::10])    
        plt.xlabel("Wavelength")    
        plt.ylabel("F Value")    
      
        """PLOT BAR ANOTHER WAY
        uni=pd.DataFrame(univariate)
        plt.figure()
        ax=uni[uni.columns[0]].plot(kind="bar",title=" hey ",figsize=(20,8),legend=True)
        plt.show()"""
        k_best_features = SelectKBest(f_classif, k=int(self.threshold)).fit(X_train.fillna(0), y_train)
        xtrain=X_train.columns[k_best_features.get_support()]
        self.k_Best=pd.DataFrame(xtrain)
        xy=X_train.columns[k_best_features.get_support()]
        
    def ANOVA(self):
        self.threshold=self.lineEdit.text()
        df1 = pd.read_csv(self.filepath[0])
        df2 = pd.read_csv(self.filepath[1])
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df1.select_dtypes(include=numerics).columns)
        data1 = df1[numerical_features]
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df2.select_dtypes(include=numerics).columns)
        data2 = df2[numerical_features]
        data1=data1.transpose()
        data2=data2.transpose()
        out=pd.DataFrame(columns=['F val','P val','K Best'])
        #print(out)
        for i in range(len(data1)):
            f,p=stats.f_oneway(data1.iloc[i,:],data2.iloc[i,:])
            out.loc[i,'F val']=f
            out.loc[i,'P val']=p
        out.index=data1.index    
        out['K Best']=0
        alpha=float(self.lineEdit_2.text())
        output=out.sort_values('F val',ascending=False,inplace=False)
        count=0
        for i in range(len(out)):
            if(output.iloc[i,1]<alpha and count<int(self.threshold)):
                print(output.iloc[i,1])
                output.iloc[i,2]=1
                count=count+1
        if(count<int(self.threshold)):
            messageDisplay="Only "+str(count)+" values selected from total of "+self.threshold+" values."
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
        output.index=output.index.astype(float)
        output.sort_index(inplace=True)    
        self.output1=output
        self.output1.to_csv(self.outputFilename)
        ax2=output['F val'].plot(kind='bar',color=np.where(output['K Best']==1.0,'g','r'))
        ticks=(ax2.xaxis.get_ticklocs())
        ticklabels=[n.get_text() for n in ax2.xaxis.get_ticklabels()]
        for i in range(0,len(output)):
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
        plt.xlabel("Wavelength")
        plt.ylabel("F Value")
        
    def kruskal(self):
        self.threshold=self.lineEdit.text()
        df1 = pd.read_csv(self.filepath[0])
        df2 = pd.read_csv(self.filepath[1])
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df1.select_dtypes(include=numerics).columns)
        data1 = df1[numerical_features]
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df2.select_dtypes(include=numerics).columns)
        data2 = df2[numerical_features]
        data1=data1.transpose()
        data2=data2.transpose()
        out=pd.DataFrame(columns=['H val','P val','K Best'])
        print(out)
        for i in range(len(data1)):
            f,p=stats.kruskal(data1.iloc[i,:],data2.iloc[i,:])
            out.loc[i,'H val']=f
            out.loc[i,'P val']=p
        out.index=data1.index    
        out['K Best']=0
        alpha=float(self.lineEdit_2.text())
        output=out.sort_values('H val',ascending=False,inplace=False)
        count=0
        for i in range(len(out)):
            if(output.iloc[i,1]<alpha and count<int(self.threshold)):
                output.iloc[i,2]=1
                count=count+1
        if(count<int(self.threshold)):
            messageDisplay="Only "+str(count)+" values selected from total of "+self.threshold+" values."
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)

        output.index=output.index.astype(float)
        output.sort_index(inplace=True)    
        #print(output)
        self.output1=output
        self.output1.to_csv(self.outputFilename)
        ax2=output['H val'].plot(kind='bar',color=np.where(output['K Best']==1.0,'g','r'))
        ticks=(ax2.xaxis.get_ticklocs())
        ticklabels=[n.get_text() for n in ax2.xaxis.get_ticklabels()]
        for i in range(0,len(output)):
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
        plt.xlabel("Wavelength")
        plt.ylabel("H Value")
    
    def SDI(self):
        df1=pd.read_csv(self.filepath[0])
        df2=pd.read_csv(self.filepath[1])
        threshold=float(self.lineEdit.text())
        print(threshold)
        sdi=(abs(df1.mean()-df2.mean()))/(df1.std()+df2.std())
        sdi=pd.DataFrame(sdi)
        sdi['Best']=1
        output=sdi.sort_values(sdi.columns[0],ascending=False,inplace=False)
        count=0
        for j in range(len(output)):
            if(output.iloc[j,0]<threshold):
                output.iloc[j,1]=0
                count=count+1
        if(count==0):
            messageDisplay="No values selected"
            QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)

        output.index=output.index.astype(float)
        output.sort_index(inplace=True)     
        self.output1=output
        self.output1.to_csv(self.outputFilename)
        ax=output[output.columns[0]].plot(kind='bar',color=np.where(output['Best']<threshold,'g','r'))
        ticks=(ax.xaxis.get_ticklocs())
        ticklabels=[n.get_text() for n in ax.xaxis.get_ticklabels()]
        for i in range(0,len(sdi)):
            ticklabels[i]=float(("%.2f")%float(ticklabels[i]))
        if(i>1000):
            ax.xaxis.set_ticks(ticks[::100])
            ax.xaxis.set_ticklabels(ticklabels[::100])
        elif(i<1000 and i>500):
            ax.xaxis.set_ticks(ticks[::50])
            ax.xaxis.set_ticklabels(ticklabels[::50])
        else:
            ax.xaxis.set_ticks(ticks[::10])
            ax.xaxis.set_ticklabels(ticklabels[::10])
        plt.xlabel("Wavelength")
        plt.ylabel("SDI")
#        ax.xaxis.set_ticks(ticks[::10])
#        ax.xaxis.set_ticklabels(ticklabels[::10])
        
    def Normalised_Index_Selection(self):
        
        def update_progress(job_title, progress):
            length = 20 # modify this to change the length
            block = int(round(length*progress))
            msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
            if progress >= 1: msg += " DONE\r\n"
            sys.stdout.write(msg)
            sys.stdout.flush()   
        df1=pd.read_csv(self.filepath[0])
        df2=pd.read_csv(self.filepath[1])
        l3=df1.append(df2,ignore_index=False)
        df3=pd.DataFrame(l3)
        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
        numerical_features = list(df3.select_dtypes(include=numerics).columns)
        X = df3[numerical_features]
#        print("HI after X")
        SDI=np.zeros((372,372))
        for i in range(0,372):
            update_progress("Normalized Index", (i/float(372)))

            for j in range(0,372):
                if i==j:
                    SDI[i,j]=0
                else:
                    a=np.mean(X.iloc[:,max(i-2,0):min(i+2,372)],axis=1)-np.mean(X.iloc[:,max(j-2,0):min(j+2,372)],axis=1)
                    b=np.mean(X.iloc[:,max(i-2,0):min(i+2,372)],axis=1)+np.mean(X.iloc[:,max(j-2,0):min(j+2,372)],axis=1)
                    c=a/(b+.0001)
                    m1=np.mean(c.iloc[0:19])
                    m2=np.mean(c.iloc[20:40])
                    s1=np.std(c.iloc[0:19])
                    s2=np.std(c.iloc[20:40])
                    SDI[i,j]=np.abs(m1-m2)/(s1+s2)
        update_progress("Normalized Index", (1.0))

        output=pd.DataFrame(SDI)
        output.index=X.transpose().index
        output=output.transpose()
        output.index=X.transpose().index
        self.output1=output
        self.output1.to_csv(self.outputFilename)

        fig = plt.figure()
        ax2 = plt.subplot(111) 
        ax2.imshow(output, interpolation="none")

        ticklabels=list(output.index[::45])
        for i in range(0,len(ticklabels)):
            ticklabels[i]=float(("%.1f")%float(ticklabels[i]))
        tick = np.arange(0, output.index.shape[0], 45)
        plt.xticks(tick, ticklabels)
        plt.yticks(tick, ticklabels)
        plt.title("Normalised Index")

#    def JMDistance(self):
#        df1=pd.read_csv(self.filepath[0])
#        df2=pd.read_csv(self.filepath[1])
#        numerics = ['int16', 'int32','int64', 'float16', 'float32', 'float64']
#        numerical_features = list(df1.select_dtypes(include=numerics).columns)
#        X = df1[numerical_features]
#        numerical_features = list(df2.select_dtypes(include=numerics).columns)
#        Y = df2[numerical_features]
#        covx=np.cov(X)
#        covy=np.cov(Y)
#        internal=(np.linalg.det((covx+covy)/2))/(np.sqrt(abs(np.linalg.det(covx)))*np.sqrt(abs(np.linalg.det(covy))))
#        part5=(0.5)*np.log(internal)
#        part1=(np.linalg.inv((covx+covy)/2))
#        part2=(X-Y).T
#        part3=np.dot(part2,part1)
#        part4=(1/8)*np.dot(part3,(X-Y))
#        np.log(2)
#        negB=-1*(part4+part5)
#        J=2.0*(1.0-np.exp(negB))
#        print(J)
#        self.output1=pd.DataFrame(J)
#        self.output1.to_csv(self.outputFilename)
        
      
    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()

        self.outputFilename,_=QFileDialog.getSaveFileName(None,'save',lastDataDir,'*.csv')


        if not self.outputFilename:
            return

        if self.outputFilename:
            self.lineEdit_4.setText(self.outputFilename)

        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename
    
    
    def run(self):
        if(self.lineEdit_3.text()==""):
                messageDisplay="Cannot leave Input empty!"
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return
        filepath=str(self.lineEdit_3.text()).split()
        for i in filepath:         
            if(str(i.split('/')[-1].split('.')[-1])=="csv"):
                pass
            else:
                self.lineEdit_3.setFocus()
                self.lineEdit_3.selectAll()
                messageDisplay="Input file extension cannot be "+str(i.split('/')[-1].split('.')[1])
                QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                return 
        c=0
        count=0
        print(len(filepath))
        for i in range(len(filepath)):
                if(len(self.filepath)==0):
                    print("Whatever")
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
        print(count)
        print(i)
        if(count==i+1):
            self.filepath=filepath
        print(self.filepath)
        if(c==1):
                self.lineEdit_3.setFocus()
                self.lineEdit_3.selectAll()
                
                for i in filepath:
                    print(i.rsplit('/',1)[0])
                    print(i)
                    if (path.exists(i.rsplit('/',1)[0])==True):
                        pass
                    else:
                        messageDisplay="Path does not exist!"
                        QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                        return
                    if (path.isfile(i)==True):
                        pass
                    else:
                        messageDisplay="File does not Exist!"
                        QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
                        return
                self.filepath=filepath
                
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
            self.lineEdit_4.setFocus()
            self.lineEdit_4.selectAll()
            messageDisplay="Output File Path does not exist!"
            QtWidgets.QMessageBox.information(self.Form,'Error',messageDisplay,QtWidgets.QMessageBox.Ok)
            return
        self.outputFilename=outputPath[0]
        splitVal1=self.filepath[0].rsplit('.',1)
        splitVal2=self.filepath[1].rsplit('.',1)
        
        
        if(splitVal1[1]==splitVal2[1]=="csv"):
            if(self.comboBox.currentText()=="Mutual Information"):
                if(self.lineEdit.text()==""):
                    messageDisplay="Threshold value cannot be left empty!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
                try:
                    int(self.lineEdit.text())
                except Exception:
                    messageDisplay="Threshold can only be of Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
                self.mutualInformation()
                
            if(self.comboBox.currentText()=="ANOVA"):
                if(self.lineEdit.text()==""):
                    messageDisplay="Threshold value cannot be left empty!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
                if(self.lineEdit_2.text()==""):
                    messageDisplay="Alpha cannot be left empty!"
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
                
            if(self.comboBox.currentText()=="F Classification"):
                if(self.lineEdit.text()==""):
                    messageDisplay="Threshold value cannot be left empty!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
                try:
                    int(self.lineEdit.text())
                except Exception:
                    messageDisplay="Threshold can only be of Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
                self.fclassification()
                
            if(self.comboBox.currentText()=="Kruskal"):
                if(self.lineEdit.text()==""):
                    messageDisplay="Threshold value cannot be left empty!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
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
                if(self.lineEdit.text()==""):
                    messageDisplay="Threshold value cannot be left empty!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
                try:
                    float(self.lineEdit.text())
                except Exception:
                    messageDisplay="Threshold can only be of Float or Integer type!"
                    QtWidgets.QMessageBox.information(self.Form,'Message',messageDisplay,QtWidgets.QMessageBox.Ok)
                    return
                self.SDI()
                
            if(self.comboBox.currentText()=="Normalised Index Selection"):
                self.Normalised_Index_Selection()
#            if(self.comboBox.currentText()=="JM Distance"):
#                self.JMDistance()
                

        
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    Form = QWidget()
    #QSizePolicy sretain=Form.sizePolicy()
    #sretain.setRetainSizeWhenHidden(True)
    #sretain.setSizePolicy()
    ui = FeatureSelection()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
