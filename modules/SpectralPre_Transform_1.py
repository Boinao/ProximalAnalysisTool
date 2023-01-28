from PyQt5 import QtWidgets
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5 import QtWidgets
from sklearn.metrics import mean_squared_error, r2_score
from Ui.SpectralPre_TransformUi import Ui_Form
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from scipy import stats
from sklearn import preprocessing
import scipy.signal._savitzky_golay as savgol
# from . import GdalTools_utils as Utils
POSTFIX = '_Preprocess_Transform'
from modules import Utils
import csv as pycsv

class Preprocess_Transoform(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.outputFilename = ""
        self.data = pd.DataFrame()

    def get_widget(self):
        return self.groupBox

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def setupUi(self, Form):
        super(Preprocess_Transoform, self).setupUi(Form)
        self.Form = Form

        self.radioButton_5.setChecked(True)

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())

    def SpectrabrowseButton_clicked(self):
        fname = []
        # if self.curdir is None:
        #     self.curdir = os.getcwd()
        #     self.curdir = self.curdir.replace("\\", "/")
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_2.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            return


        self.filepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit_2.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))
            self.outputFilename = os.path.dirname(fname) + "/Output" + POSTFIX + ".csv"
            self.lineEdit.setText(self.outputFilename)
        else:
            self.lineEdit_2.setText("")

    def saveas(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')

        if not self.outputFilename:
            return

        self.lineEdit_3.setText(self.outputFilename)

        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))
        return self.outputFilename

    def Soil_property_plot(self):
        self.mplWidgetSpectral_5.ax.clear()
        for i in range(0, len(self.sparam)):

            self.mplWidgetSpectral_5.ax.hist(self.soilp[:, i], 15, label=str(self.sparam[i]))
            #    plt.xlabel(str(sparam[i]))
            self.mplWidgetSpectral_5.ax.legend()
            self.mplWidgetSpectral_5.ax.set_title('Distribution of' + str(self.sparam[i]))

        self.mplWidgetSpectral_5.canvas.draw()
        self.soilp.tofile(self.outputFilename,sep = ',')

    def Transform_Box_Cox_plot(self):
        self.mplWidgetSpectral_5.ax.clear()
        boxcox_clay = stats.boxcox(self.soilp[:, 0])[0]
        boxcox_oc = stats.boxcox(self.soilp[:, 1])[0]
        boxcox_iron = stats.boxcox(self.soilp[:, 2])[0]
        boxcox_soil = np.vstack((boxcox_clay, boxcox_oc, boxcox_iron)).T
        for i in range(0, len(self.sparam)):
            plt.subplot(4, 3, i + 4)
            plt.hist(boxcox_soil[:, i], 15,label=str(self.sparam[i]))
            #    plt.xlabel(str(sparam[i]))
            plt.legend()
            plt.title('Box-Cox transformed' + str(self.sparam[i]))
        plt.show()
        boxcox_soil.to_file(self.outputFilename,sep=',')

    def Transform_log_plot(self):
        log_clay = np.log(self.soilp[:, 0])
        log_oc = np.log(self.soilp[:, 1])
        log_iron = np.log(self.soilp[:, 2])
        log_soil = np.vstack((log_clay, log_oc, log_iron)).T
        for i in range(0, len(self.sparam)):
            plt.subplot(4, 3, i + 7)
            plt.hist(log_soil[:, i], 15,label=str(self.sparam[i]))
            #    plt.xlabel(str(sparam[i]))
            plt.legend()
            plt.title('Log transformed' + str(self.sparam[i]))
        plt.show()
        log_soil.to_file(self.outputFilename,sep=',')

    def Transform_standard_plot(self):

        scale_clay = stats.zscore(self.soilp[:, 0])
        scale_oc = stats.zscore(self.soilp[:, 1])
        scale_iron = stats.zscore(self.soilp[:, 2])  # scale_clay.mean_  , scale_clay.transform()
        scale_soil = np.vstack((scale_clay, scale_oc, scale_iron)).T
        for i in range(0, len(self.sparam)):
            plt.subplot(4, 3, i + 10)
            plt.hist(scale_soil[:, i], 15,label=str(self.sparam[i]))
            plt.xlabel(str(self.sparam[i]))
            plt.legend()
            plt.title('Zscore transformed' + str(self.sparam[i]))
        #    plt.subplot.atight_layout(pad=.1,w_pad=.1,h_pad=.1)
        plt.tight_layout()
        plt.show()
        scale_soil.T.to_csv(self.outputFilename)

    def Plot_spectra_plot(self):
        '''-----    PLOT SPECTRA----- '''
        steps = 250
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(self.wavelength, self.spectra, label='Soil Spectra')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.ylim([0, .9])
        plt.xlim([350, 2500])
        plt.yticks(np.arange(0, .9, .1))
        plt.xticks(self.wavelength[::steps])
        # plt.legend()
        plt.title('Proximal-Soil-Spectra')

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
    def Spectral_Transform_log(self):
        self.mplWidgetSpectral_6.ax.clear()
        log_spec = 1 / np.log(self.spectra)
        self.mplWidgetSpectral_6.ax.plot(self.wavelength, log_spec, label='Log transformed')
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel('Absorbance')
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(self.wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_title('Log transformed Spectra')
        self.mplWidgetSpectral_6.canvas.draw()
        log_spec.T.to_csv(self.outputFilename)

    def Spectral_transform_standard_plot(self):
        self.mplWidgetSpectral_6.ax.clear()
        scale = preprocessing.StandardScaler()
        # scale_clay.mean_  , scale_clay.transform()
        scale_fit = scale.fit(self.spectra.T)
        scale_spec = scale_fit.transform(self.spectra.T).T
        self.mplWidgetSpectral_6.ax.plot(self.wavelength, scale_spec, label='Standard Scalar transformed')
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel('zscore trransforemd')
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(self.wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_title('Standard Scalar transformed Spectra')
        self.mplWidgetSpectral_6.canvas.draw()

    def SpectralTransformStandard_Normal(self):
        self.mplWidgetSpectral_6.ax.clear()
        scale = preprocessing.StandardScaler()
        # scale_clay.mean_  , scale_clay.transform()
        scale_fit = scale.fit(self.spectra)
        scalen_spec = scale_fit.transform(self.spectra)
        self.mplWidgetSpectral_6.ax.plot(self.wavelength, scalen_spec, label='Standard Normal Variate')
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel('zscore trransforemd')
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(self.wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_title('Standard Normal Variate transformed Spectra')
        self.mplWidgetSpectral_6.canvas.draw()
    def Spectral_Derivative(self):

        self.mplWidgetSpectral_6.ax.clear()
        sav_spec = savgol.savgol_filter(self.spectra, 9, 2, deriv=1, axis=0, mode='nearest')  # Derivative spectra
        self.mplWidgetSpectral_6.ax.plot(self.wavelength, sav_spec, label='1st Derivative')
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel('Values')
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(self.wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_title('1st Derivative Spectra')
        self.mplWidgetSpectral_6.canvas.draw()
    def SpectralSmoothing_Savitsky_Golay(self):
        self.mplWidgetSpectral_6.ax.clear()
        sav_spec = savgol.savgol_filter(self.spectra, 9, 2, deriv=0, axis=0, mode='nearest')  # Derivative spectra
        self.mplWidgetSpectral_6.ax.plot(self.wavelength, sav_spec, label='Savitsky-Golay Filtered')
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel('Savitsky-Golay')
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(self.wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_title('Savitsky-Golay Filtered Spectra')
        self.mplWidgetSpectral_6.canvas.draw()
    def Wavelength_correlation(self):
        self.mplWidgetSpectral_6.ax.clear()
        spec = self.spectra.T
        CM = np.zeros((3, self.row))
        for i in range(0, len(self.sparam)):
            for j in range(0, self.row):
                CM[i, j] = np.corrcoef(spec[:, j], self.soilp[:, i])[0][1]

        labels = 'Clay', 'OC', 'Iron'

        [a, b, c] = self.mplWidgetSpectral_6.ax.plot(self.wavelength,CM.T, label=labels)
        self.mplWidgetSpectral_6.ax.legend([a, b, c], labels, loc=1)
        self.mplWidgetSpectral_6.ax.set_xlabel('Wavelength (nm)')
        self.mplWidgetSpectral_6.ax.set_ylabel('Correlation Coefficient')
        self.mplWidgetSpectral_6.ax.set_xlim([350, 2500])
        self.mplWidgetSpectral_6.ax.set_xticks(self.wavelength[::250])
        self.mplWidgetSpectral_6.ax.set_ylim([-1, 1.2])
        self.mplWidgetSpectral_6.ax.set_title('Correlation of Soil-Properties vs wavelength')
        self.mplWidgetSpectral_6.canvas.draw()


    def isHeaderLine(self,line):
        """
        Returns True if str ``line`` could be a CSV header
        :param line: str
        :return: str with CSV dialect
        """
        for dialect in [pycsv.excel_tab, pycsv.excel]:
            fieldNames = [n.lower() for n in pycsv.DictReader([line], dialect=dialect).fieldnames]
            for column in ['wavelength', 'Wavelength']:
                if column in fieldNames:
                    return dialect
        return None

    def canRead(self,path=None):
        if not isinstance(path, str):
            return False

        if not os.path.isfile(path):
            return False

        # mbytes = os.path.getsize(path) / 1000 ** 2
        # if mbytes > MAX_CSV_SIZE:
        #     return False

        try:

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line) > 0 and not line.startswith('#'):
                        dialect = self.isHeaderLine(line)
                        if dialect is not None:
                            return dialect
        except Exception as ex:
            return None
        return None

    def run(self):

        if (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            messageDisplay = "Cannot leave Input empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
            self.lineEdit.setFocus()
            messageDisplay = "Cannot leave Output empty!"
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_2.text())):
            self.lineEdit_2.setFocus()
            messageDisplay = "Path does not exist : " + self.lineEdit_2.text()
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.basename(self.lineEdit_2.text()).split('.')[-1]=='csv'):
            self.lineEdit_2.setFocus()
            messageDisplay = "File not supported : " + os.path.basename(self.lineEdit_2.text()).split('.')[-1]
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.isdir(os.path.dirname(self.lineEdit.text()))):
            self.lineEdit.setFocus()
            messageDisplay ="Kindly enter a valid output path."
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return


        if not self.canRead(self.lineEdit_2.text()):
            self.lineEdit_2.setFocus()
            print(self.canRead(self.lineEdit_2.text()))
            messageDisplay = "Invalid CSV format, data should contain Columns : Wavelength, spectra1, spectra2,... "
            QtWidgets.QMessageBox.information(self.Form, 'Error', messageDisplay,
                                              QtWidgets.QMessageBox.Ok)
            return



        try:
            data = pd.read_csv(self.filepath, header=0, index_col=0)
            self.df_spectra=data.T
            wave = list(self.df_spectra.columns.ravel()[4:])
            samples, cols = self.df_spectra.shape
            bands = cols - 4

            spectra = np.array(self.df_spectra.iloc[:, 4:cols])  # read Spectra
            spectra = spectra.T
            self.wavelength = np.array(wave, dtype=int)  # read wavelength
            # ---- Read soil properties---------------#
            Clay = np.array(self.df_spectra.iloc[:, 1])
            OC = np.array(self.df_spectra.iloc[:, 2])
            Iron = np.array(self.df_spectra.iloc[:, 3])
            soilp = np.vstack((Clay, OC, Iron)).T
            # ---- Final data--------------------#
            indx = [soilp[:, 0] > 0][0] * 1
            indx = np.nonzero(indx)
            spectra = spectra[:, indx[0]]
            self.soilp = soilp[indx[0], :]
            self.row, col = spectra.shape

            '''----------- Remove offset/Jump Error-------------'''
            S = spectra.copy()
            SS = np.zeros((self.row, col))
            SS = S

            a = S[650, :] - S[651, :]
            b = S[651::, :]
            r, t = b.shape
            import numpy.matlib

            rep = np.matlib.repmat(a, r, 1)
            SS[651::, :] = rep + b

            a = SS[1480, :] - SS[1481, :]
            b = S[1481:, :]
            r, t = b.shape
            import numpy.matlib

            rep = np.matlib.repmat(a, r, 1)
            SS[1481:, :] = rep + b
            self.spectra = SS

            '''------------ Spectral Stistics--------------'''
            self.mean = np.mean(self.spectra, axis=1)
            self.std = np.std(self.spectra, axis=1)

            '''---- PLOTS Property HISTOGRAM------'''
            self.sparam = ['Clay', 'Organic Carbon', 'Iron']
            if self.Soil_property.isChecked():
                self.Soil_property_plot()
            if self.Transform_box_cox.isChecked():
                self.Transform_Box_Cox_plot()
            if self.Transform_log.isChecked():
                self.Transform_log_plot()
            if self.Transform_standard.isChecked():
                self.Transform_standard_plot()
            if self.Plot_spectra.isChecked():
                self.Plot_spectra_plot()
            if self.radioButton_5.isChecked():
                self.Spectral_Transform_log()
            if self.Spectral_transform_standard.isChecked():
                self.Spectral_transform_standard_plot()
            if self.radioButton.isChecked():
                self.SpectralTransformStandard_Normal()
            if self.radioButton_2.isChecked():
                self.Spectral_Derivative()
            if self.radioButton_3.isChecked():
                self.SpectralSmoothing_Savitsky_Golay()
            if self.radioButton_4.isChecked():
                self.Wavelength_correlation()


        except Exception as e:
            import traceback
            print(e, traceback.format_exc())
