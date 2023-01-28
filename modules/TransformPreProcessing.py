
from PyQt5 import QtWidgets
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5 import QtWidgets
from sklearn.metrics import mean_squared_error, r2_score
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from scipy import stats
from sklearn import preprocessing
import scipy.signal._savitzky_golay as savgol
import seaborn as sns
from Ui.SpectralPre_TransformUi import Ui_Form

# from . import GdalTools_utils as Utils
POSTFIX = '_Preprocess_Transform'
from modules import Utils
import csv as pycsv
import pywt

class Preprocess_Transoform(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.outputFilename = ""
        self.data = pd.DataFrame()

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
        super(Preprocess_Transoform, self).setupUi(Form)
        self.Form = Form
        self.tabWidget_4.setCurrentIndex(0)

        self.connectWidgets()
        self.toggleRadio_property()

    def populateWvletFamily(self):
        self.wvFamilyCmb.clear()

        self.wvFamilyCmb.addItem("--Select--")
        self.wvFamilyCmb.addItem("haar")
        self.wvFamilyCmb.addItem("db")
        self.wvFamilyCmb.addItem("bior")
        self.wvFamilyCmb.addItem("coif")
        self.wvFamilyCmb.addItem("dmey")
        self.wvFamilyCmb.addItem("rbio")
        self.wvFamilyCmb.addItem("sym")

        self.wvFamilyCmb.currentIndexChanged.connect(self.updateWaveletCmb)



    def connectWidgets(self):
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_5.clicked.connect(lambda: self.MetadatabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())

        self.togglePropertyChk.clicked.connect(self.toggleRadio_property)
        self.populateWvletFamily()




        # self.property.toggled.connect(self.toggleRadio_property)
    def updateWaveletCmb(self,i):
        self.WvCmb.clear()
        self.WvCmb.addItem("--Select--")

        if i==0:
            return
        waveletf = self.wvFamilyCmb.currentText()
        wavelet_name=pywt.wavelist(waveletf)

        for item in wavelet_name:
            self.WvCmb.addItem(str(item))

    def toggleRadio_property(self):
        page=self.tabWidget_4.widget(1)
        if page is not None:
            page.setEnabled(self.togglePropertyChk.isChecked())
        self.lineEdit.setEnabled(self.togglePropertyChk.isChecked())
        self.pushButton_5.setEnabled(self.togglePropertyChk.isChecked())



        # self.comboBox.setEnabled(self.togglePropertyChk.isChecked())

    def SpectrabrowseButton_clicked(self):

        fname = Utils.browseInputFile(POSTFIX + ".csv", self.lineEdit_2, "Supported types (*.csv)", self.lineEdit_3)

        # print(self.filepath)
        if fname:
            if not Utils.validateInputFormat(fname):
                self.lineEdit_2.setText("")
                self.lineEdit_3.setText("")
                return
        else:
            self.lineEdit_2.setText("")
            self.lineEdit_3.setText("")

    def populateProperty(self):
        self.metafilepath = self.lineEdit.text()
        df_metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
        self.data = df_metadata.T

        self.comboBox.clear()
        self.comboBox.addItem("--Select--")
        value = self.data.drop(["Spectra ID", "class_label", "class"], axis=1)
        for i in value.columns:
            self.comboBox.addItem(str(i))

    def MetadatabrowseButton_clicked(self):

        fname = Utils.browseMetadataFile(self.lineEdit, "Supported types (*.csv)")
        if fname:
            if not Utils.validateMetaFormat(self.lineEdit_2.text(), fname):
                self.lineEdit.setText("")
                return
            self.populateProperty()
        else:
            self.lineEdit.setText("")
            self.comboBox.clear()


    def saveas(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')

        if not self.outputFilename:
            return

        self.lineEdit_3.setText(self.outputFilename)

        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))
        return self.outputFilename


    def plotProperty(self):
        self.mplWidgetSpectral_5.ax.clear()
        label_one = str(self.comboBox.currentText())
        try:
            property_name = np.array(self.df_metadata[label_one]).astype('float')
        except Exception as e:
            QtWidgets.QMessageBox.information(self.Form, 'Error', "Error :"+str(e),
                                              QtWidgets.QMessageBox.Ok)
            return
        if self.radioButton_8.isChecked():
            value = stats.boxcox(property_name)[0]
            type_prop='_boxcox'

        if self.radioButton_9.isChecked():
            value = np.log(property_name)
            type_prop = '_log'
        if self.radioButton_10.isChecked():
            value=property_name
            type_prop = '_none'


        sns.distplot(a=value, label=str(label_one), ax=self.mplWidgetSpectral_5.ax)
        self.mplWidgetSpectral_5.ax.legend()
        self.mplWidgetSpectral_5.canvas.draw()
        df1 = pd.DataFrame(value)
        df1.to_csv(self.outputFilename + type_prop + '.csv', header=True)
        self.mplWidgetSpectral_5.canvas.draw()

    def Plot_spectra_plot(self):
        '''-----    PLOT SPECTRA----- '''
        steps = 250
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 2)
        plt.errorbar(self.wavelength[::10], self.mean[::10], yerr=self.std[::10], fmt='.-', capsize=0,
                     label='Standard Deviation')
        plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.ylim([0, .9])
        plt.yticks(np.arange(0, .9, .1))
        plt.xlim([350, 2500])
        plt.xticks(self.wavelength[::steps])
        plt.title('Mean Spectra and Standard-Deviation')
        plt.show()


    def plotTransform(self,wavelength,df_spectra, transform='log'):
        self.mplWidgetSpectral_6.ax.clear()

        if transform=='log':
            log_spec = 1 / np.log(df_spectra)
            values=log_spec.T
            suffix='_Spectra_Transform_log' + '.csv'
            label='Log transformed'
            ylabel='Absorbance'
            title='Log transformed Spectra'
        elif transform=='std':
            spectra = df_spectra.T
            scale = preprocessing.StandardScaler()
            scale_fit = scale.fit(spectra.T)
            scale_spec = scale_fit.transform(spectra.T)
            values=scale_spec.T


            suffix = '_Spectra_Transform_standard' + '.csv'
            label='Standard Scalar transformed'
            ylabel = 'zscore trransformed'
            title = 'Standard Scalar transformed Spectra'

        elif transform=='norm':
            spectra=df_spectra.T
            scale = preprocessing.StandardScaler()
            scale_fit = scale.fit(spectra)
            values = scale_fit.transform(spectra)

            suffix = '_Spectra_Transform_standard_normal' + '.csv'
            label = 'Standard Normal Variate'
            ylabel = 'zscore transformed'
            title = 'Standard Normal Variate Spectra'

        elif transform == 'deri':
            values = np.diff(self.df_spectra.T)  # Derivative spectra

            suffix = '_Spectra_Transform_derivative' + '.csv'
            label = '1st Derivative'
            ylabel = 'Values'
            title = '1st Derivative Spectra'

        elif transform=='savi':
            sav_spec = savgol.savgol_filter(self.df_spectra, 9, 2, deriv=0, axis=0,
                                            mode='nearest')  # Derivative spectra
            values=sav_spec.T
            suffix = '_Spectra_Transform_Savitsky_Golay' + '.csv'
            label = 'Savitsky-Golay Filtered'
            ylabel = 'Savitsky-Golay'
            title = 'Savitsky-Golay Filtered Spectra'


        self.mplWidgetSpectral_6.ax.plot(wavelength, values, label=label)
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel(ylabel)
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_title(title)
        self.mplWidgetSpectral_6.canvas.draw()
        df1 = pd.DataFrame(values, index=wavelength)
        df1.to_csv(self.outputFilename + suffix, header=True, index=True)


    def Wavelength_correlation(self):
        self.mplWidgetSpectral_6.ax.clear()
        spec = self.df_spectra.T
        CM = np.zeros((3, self.row))
        for i in range(0, len(self.sparam)):
            for j in range(0, self.row):
                CM[i, j] = np.corrcoef(spec[:, j], self.soilp[:, i])[0][1]

        labels = 'Clay', 'OC', 'Iron'

        [a, b, c] = self.mplWidgetSpectral_6.ax.plot(self.wavelength, CM.T, label=labels)
        self.mplWidgetSpectral_6.ax.legend([a, b, c], labels, loc=1)
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel('Correlation Coefficient')
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(self.wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_ylim([-1, 1.2])
        self.mplWidgetSpectral_6.ax.set_title('Correlation of Soil-Properties vs wavelength')
        self.mplWidgetSpectral_6.canvas.draw()
        df1 = pd.DataFrame(CM.T, index=self.wavelength)
        df1.to_csv(self.outputFilename + '_Spectra_Transform_Wavelength_correlation' + '.csv', header=True, index=True)



    def run(self):

        self.mplWidgetSpectral_5.clear()
        self.mplWidgetSpectral_6.clear()
        if (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            messageDisplay = "Cannot leave Input empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_3.text() is None) or (self.lineEdit_3.text() == ""):
            self.lineEdit_3.setFocus()
            messageDisplay = "Cannot leave Output empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_2.text())):
            self.lineEdit_2.setFocus()
            messageDisplay = "Path does not exist : " + self.lineEdit_2.text()
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.basename(self.lineEdit_2.text()).split('.')[-1] == 'csv'):
            self.lineEdit_2.setFocus()
            messageDisplay = "File not supported : " + os.path.basename(self.lineEdit_2.text()).split('.')[-1]
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return


        if (not os.path.isdir(os.path.dirname(self.lineEdit_3.text()))):
            self.lineEdit_3.setFocus()
            messageDisplay = "Kindly enter a valid output path."
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return


        fname=self.lineEdit_2.text()
        if not Utils.validateInputFormat(fname):
            return


        if self.tabWidget_4.currentIndex()==1 and (self.comboBox.currentText()=='--Select--'):
            self.comboBox.setFocus()
            messageDisplay = "Please select any one of the property first "
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay,
                                              QtWidgets.QMessageBox.Ok)
            return



        try:
            self.outputFilename = self.lineEdit_3.text()
            self.inFile = self.lineEdit_2.text()
            if self.togglePropertyChk.isChecked():
                if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
                    self.lineEdit.setFocus()
                    messageDisplay = "Input missing ! "
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay,
                                                      QtWidgets.QMessageBox.Ok)
                    return
                else:
                    self.Metafile = self.lineEdit.text()
            else:
                self.Metafile = 'Not Selected!'



            print("In: " + self.inFile)
            print("Out: " + self.outputFilename)
            print("Meta data: " + self.Metafile)
            print("Running...")

            data = pd.read_csv(self.inFile, header=0, index_col=None)
            spectra = data.iloc[:, 1:].to_numpy()

            self.df_spectra = spectra.T
            if self.togglePropertyChk.isChecked():
                metadata = pd.read_csv(self.Metafile, header=None, index_col=0)
                self.df_metadata = metadata.T

            wavelength = data.iloc[:, 0].to_numpy()


            self.wavelength = wavelength.T

            # '''------------ Spectral Stistics--------------'''
            self.mean = self.df_spectra.mean(axis=1)
            self.std = self.df_spectra.mean(axis=1)

            '''---- PLOTS Property HISTOGRAM------'''
            self.sparam = ['Clay', 'Organic Carbon', 'Iron']
            self.row, self.col = self.df_spectra.shape


            if self.tabWidget_4.currentIndex()==0:
                # if self.Plot_spectra.isChecked():
                #     self.Plot_spectra_plot()
                if self.radioButton_5.isChecked():
                    self.plotTransform(self.wavelength, self.df_spectra, 'log')
                if self.Spectral_transform_standard.isChecked():
                    self.plotTransform(self.wavelength,self.df_spectra,'std')
                if self.radioButton.isChecked():
                    self.plotTransform(self.wavelength, self.df_spectra, 'norm')
                if self.radioButton_2.isChecked():
                    self.plotTransform(self.wavelength, self.df_spectra, 'deri')
                if self.radioButton_3.isChecked():
                    self.plotTransform(self.wavelength, self.df_spectra, 'savi')

            elif self.tabWidget_4.currentIndex()==1:

                if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
                    self.lineEdit.setFocus()
                    messageDisplay = "Cannot leave Property empty!"
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
                    return

                # if self.togglePropertyChk.isChecked():
                if not Utils.validateMetaFormat(self.lineEdit_2.text(), self.lineEdit.text()):
                    self.lineEdit.setFocus()
                    return

                self.plotProperty()

            elif self.tabWidget_4.currentIndex()==2:

                if self.wvFamilyCmb.currentText() == '--Select--':
                    self.wvFamilyCmb.setFocus()
                    messageDisplay = "Please select wavelet family first "
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay,
                                                      QtWidgets.QMessageBox.Ok)
                    return

                if self.WvCmb.currentText() == '--Select--':
                    self.WvCmb.setFocus()
                    messageDisplay = "Please select wavelet"
                    QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay,
                                                      QtWidgets.QMessageBox.Ok)
                    return

                # waveletf=self.wvFamilyCmb.currentText()
                wavelet = self.WvCmb.currentText()
                level=self.lvlDecompSpb.value()
                padding_mode=None
                if self.padSmoothRbn.isChecked():
                    padding_mode='smooth'
                elif self.padReflRbn.isChecked():
                    padding_mode='reflect'
                elif self.padSymmRbn.isChecked():
                    padding_mode='symmetric'

                from pywt import wavedec
                coeffs = wavedec(spectra.T, wavelet, level=level, mode=padding_mode)

                df1 = pd.DataFrame(coeffs[1].T)
                df1.to_csv(self.outputFilename + '_wavelet_detail_coeff' + '.csv',
                           header=['Sample No ' + str(i) for i in range(1, (coeffs[1].shape[0]) + 1)], index=True)

                #
                plt.figure()
                plt.subplot(2, int(np.ceil(level / 2)) + 1, 1)
                plt.plot(coeffs[0].T)
                plt.title('Approximation Level '+str(level))
                plt.xlabel("Wavelet Coefficients")
                plt.ylabel("Values")

                l=level
                for i in range(1, level + 1):
                    plt.subplot(2, int(np.ceil(level / 2)) + 1, i+1)
                    plt.plot(coeffs[i].T)
                    plt.title("Details Level " + str(l))
                    plt.xlabel("Wavelet Coefficients")
                    plt.ylabel("Values")
                    l=l-1

                plt.show()






        except Exception as e:
            import traceback
            print(e, traceback.format_exc())
