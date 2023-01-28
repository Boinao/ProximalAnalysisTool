import sys
import os
import re
from PyQt5 import QtCore
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
import cv2 as cv
import specdal
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from Ui.Resampling_SACUi import Ui_Form
import os
from PIL import Image
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import numpy as np
import pandas as pd
from modules.PandasModel import PandasModel
import colorama
from colorama import Fore
import matplotlib.pyplot as plt
import collections
import math
from modules import Utils

pluginPath = os.path.split(os.path.dirname(__file__))[0]
POSTFIX = '_Resampling'
from os import path


class Resampling(Ui_Form):

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
        super(Resampling, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_2.clicked.connect(lambda: self.saveasButton_clicked())
        self.pushButton_3.clicked.connect(lambda: self.centralwavelength_clicked())
        self.radioButton.toggled.connect(lambda: self.stateChanged())
        self.radioButton_2.toggled.connect(lambda: self.stateChanged())
        self.radioButton_3.toggled.connect(lambda: self.stateChanged())
        self.radioButton_4.toggled.connect(lambda: self.stateChanged())
        self.radioButton_5.toggled.connect(lambda: self.stateChanged())
        self.radioButton_6.toggled.connect(lambda: self.stateChanged())
        self.radioButton_7.toggled.connect(lambda: self.stateChanged())
        # self.radioButton_2.toggled.connect(lambda: self.stateChanged2())

    def stateChanged(self):
        self.lineEdit.setDisabled(True)
        self.pushButton_3.setDisabled(True)

    def browseButton_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()

        self.lineEdit_3.setText("")
        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.csv)", directory=lastDataDir)

        if not fname:
            self.lineEdit_3.setText("")
            return

        self.filepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit_3.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))

            self.outputFilename = (os.path.dirname(fname)) + "/Output" + POSTFIX + ".csv"
            self.lineEdit_4.setText(self.outputFilename)


    def saveasButton_clicked(self):
        lastDataDir = Utils.getLastSavedDir()
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, '*.csv')
        if not self.outputFilename:
            return

        self.lineEdit_4.setText(self.outputFilename)
        Utils.setLastSavedDir(os.path.dirname(self.outputFilename))

        return self.outputFilename

    def centralwavelength_clicked(self):
        fname = []
        lastDataDir = Utils.getLastUsedDir()
        self.lineEdit.setText("")


        fname, _ = QFileDialog.getOpenFileName(None, filter="Supported types (*.hdr)", directory=lastDataDir)

        if not fname:
            return

        self.Central_wavelength_filepath = fname

        # print(self.filepath)
        if fname:
            self.lineEdit.setText(fname)
            Utils.setLastUsedDir(os.path.dirname(fname))

        else:
            self.lineEdit.setText("")

    def prepare_data(self):
        data_df = pd.read_csv(self.filepath)
        spectra = data_df.values[:, 1::]
        in_wave = data_df.values[:, 0]
        return spectra, in_wave

    def Gisat_Hx_resample(self):
        self.widget_feature_5.ax.clear()
        filepath = os.path.join(pluginPath, 'external', 'GISAT_Hx.xls')
        SRF = pd.read_excel(filepath, sheet_name="Sheet1")
        samples, cols = SRF.shape
        SRFm = SRF.values
        mult_wave = np.array((SRF.iloc[:, 0]))
        cw = SRFm[np.argmax(SRFm[:, 1::], axis=0), 0]
        n_bands = len(cw)
        fw = np.ones((n_bands)) * 5
        spectra, in_wave = self.prepare_data()
        # ''' Resampling '''
        resampler = Utils.BandResampler(in_wave, cw, fwhm1=None, fwhm2=fw)

        result = np.zeros((len(cw), spectra.shape[1]))
        for i in range(0, spectra.shape[1]):
            result[:, i] = resampler(spectra[:, i])
        self.widget_feature_5.ax.plot(cw,result)
        self.widget_feature_5.ax.set_xlabel('Wavelength')
        self.widget_feature_5.ax.set_ylabel('Value')
        self.widget_feature_5.canvas.draw()

        df1 = pd.DataFrame(result, index=cw)
        df1.to_csv(self.outputFilename + '_resample_Gisat_hx' + '.csv', header=True, index=True)


    def computeResample1(self, filepath, sheet,sheet_meta, suffix):
        SRF = pd.read_excel(filepath, sheet_name=sheet)
        mult_wave = np.array((SRF.iloc[:, 0]))
        SRF_meta = pd.read_excel(filepath, sheet_name=sheet_meta)  # ,index_col=0)
        samples, cols = SRF_meta.shape
        cw = list(SRF_meta.iloc[:, 2])
        n_bands = len(cw)
        spectra, in_wave = self.prepare_data()
        minw = np.floor(min(in_wave))
        min_ws = np.argmin(np.abs(mult_wave - minw))
        maxw = np.floor(max(in_wave))
        max_ws = np.argmin(np.abs(mult_wave - maxw))

        SRFf = SRF.iloc[min_ws + 1:max_ws, 1:].values

        from scipy import interpolate
        x = in_wave
        resam_spec = np.zeros((SRFf.shape[0], spectra.shape[1]))
        for i in range(0, spectra.shape[1]):
            xnew = SRF.iloc[min_ws + 1:max_ws, 0].values
            y = spectra[:, i]
            f = interpolate.interp1d(x, y, fill_value=0)
            resam_spec[:, i] = f(xnew)  # use interpolation function returned by `interp1d`

        spectra1 = resam_spec.T
        indx = spectra1.shape
        sent_asd = np.zeros((indx[0], n_bands))
        for j in range(0, indx[0]):
            for i in range(0, n_bands):
                sent_asd[j, i] = np.sum(SRFf[:, i] * spectra1[j, :]) / np.sum(SRFf[:, i])
        self.widget_feature_5.ax.plot(cw, sent_asd.T)
        self.widget_feature_5.ax.set_xlabel('Wavelength')
        self.widget_feature_5.ax.set_ylabel('Value')
        self.widget_feature_5.canvas.draw()

        df1 = pd.DataFrame(sent_asd.T, index=cw)
        df1.to_csv(self.outputFilename + suffix, header=True, index=True)

    def Gisat_Mx_resample(self):
        try:
            self.widget_feature_5.ax.clear()
            filepath = os.path.join(pluginPath, 'external', 'GISAT_Mx.xls')
            # SRF = pd.read_excel(filepath, sheet_name="sheet2")
            # mult_wave = np.array((SRF.iloc[:, 0]))
            # SRF_meta = pd.read_excel(filepath, sheet_name="sheet0")  # ,index_col=0)
            # samples, cols = SRF_meta.shape
            # cw = list(SRF_meta.iloc[:, 2])
            # n_bands = len(cw)
            # spectra, in_wave = self.prepare_data()
            # minw = np.floor(min(in_wave))
            # min_ws = np.argmin(np.abs(mult_wave - minw))
            # maxw = np.floor(max(in_wave))
            # max_ws = np.argmin(np.abs(mult_wave - maxw))
            #
            # SRFf = SRF.iloc[min_ws + 1:max_ws, 1:].values
            #
            # from scipy import interpolate
            # x = in_wave
            # resam_spec = np.zeros((SRFf.shape[0], spectra.shape[1]))
            # for i in range(0, spectra.shape[1]):
            #     xnew = SRF.iloc[min_ws + 1:max_ws, 0].values
            #     y = spectra[:, i]
            #     f = interpolate.interp1d(x, y, fill_value=0)
            #     resam_spec[:, i] = f(xnew)  # use interpolation function returned by `interp1d`
            #
            # spectra1 = resam_spec.T
            # indx = spectra1.shape
            # sent_asd = np.zeros((indx[0], n_bands))
            # for j in range(0, indx[0]):
            #     for i in range(0, n_bands):
            #         sent_asd[j, i] = np.sum(SRFf[:, i] * spectra1[j, :]) / np.sum(SRFf[:, i])
            # self.widget_feature_5.ax.plot(cw,sent_asd.T)
            # self.widget_feature_5.ax.set_xlabel('Wavelength')
            # self.widget_feature_5.ax.set_ylabel('Value')
            # self.widget_feature_5.canvas.draw()
            #
            # df1 = pd.DataFrame(sent_asd.T, index=cw)
            # df1.to_csv(self.outputFilename + '_resample_Gisat_mx' + '.csv', header=True, index=True)

            suffix='_resample_Gisat_mx' + '.csv'
            self.computeResample1(filepath,"sheet2","sheet0", suffix)
        except Exception as e:
            import traceback
            print(e, traceback.format_exc())


    def sentinel2_resample(self):
        self.widget_feature_5.ax.clear()
        filepath = os.path.join(pluginPath, 'external', 'sentinel_SRF.xls')
        # SRF = pd.read_excel(filepath, sheet_name="Sheet1")
        # samples, cols = SRF.shape
        # mult_wave = np.array((SRF.iloc[:, 0]))
        # SRF_meta = pd.read_excel(filepath, sheet_name="Sheet0")  # ,index_col=0)
        # samples, cols = SRF_meta.shape
        # cw = list(SRF_meta.iloc[:, 2])
        # n_bands = len(cw)
        # spectra, in_wave = self.prepare_data()
        # minw = np.floor(min(in_wave))
        # min_ws = np.argmin(np.abs(mult_wave - minw))
        # maxw = np.floor(max(in_wave))
        # max_ws = np.argmin(np.abs(mult_wave - maxw))
        #
        # SRFf = SRF.iloc[min_ws + 1:max_ws, 1:].values
        #
        # from scipy import interpolate
        # x = in_wave
        # resam_spec = np.zeros((SRFf.shape[0], spectra.shape[1]))
        # for i in range(0, spectra.shape[1]):
        #     xnew = SRF.iloc[min_ws + 1:max_ws, 0].values
        #     y = spectra[:, i]
        #     f = interpolate.interp1d(x, y, fill_value=0)
        #     resam_spec[:, i] = f(xnew)  # use interpolation function returned by `interp1d`
        #
        # spectra1 = resam_spec.T
        # indx = spectra1.shape
        # sent_asd = np.zeros((indx[0], n_bands))
        # for j in range(0, indx[0]):
        #     for i in range(0, n_bands):
        #         sent_asd[j, i] = np.sum(SRFf[:, i] * spectra1[j, :]) / np.sum(SRFf[:, i])
        # self.widget_feature_5.ax.plot(cw,sent_asd.T)
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # self.widget_feature_5.ax.set_ylabel('Value')
        # self.widget_feature_5.canvas.draw()
        #
        #
        # df1 = pd.DataFrame(sent_asd.T, index=cw)
        # df1.to_csv(self.outputFilename + '_resample_sentinel2' + '.csv', header=True, index=True)

        suffix = '_resample_sentinel2' + '.csv'
        self.computeResample1(filepath,"Sheet1","Sheet0", suffix)


    def landsat_8_resample(self):
        self.widget_feature_5.ax.clear()
        filepath = os.path.join(pluginPath, 'external', 'Landsat8_SRF.xls')
        # SRF = pd.read_excel(filepath, sheet_name="Sheet1")
        # samples, cols = SRF.shape
        # mult_wave = np.array((SRF.iloc[:, 0]))
        # SRF_meta = pd.read_excel(filepath, sheet_name="Sheet0")  # ,index_col=0)
        # samples, cols = SRF_meta.shape
        # cw = list(SRF_meta.iloc[:, 2])
        # n_bands = len(cw)
        # spectra, in_wave = self.prepare_data()
        # minw = np.floor(min(in_wave))
        # min_ws = np.argmin(np.abs(mult_wave - minw))
        # maxw = np.floor(max(in_wave))
        # max_ws = np.argmin(np.abs(mult_wave - maxw))
        #
        # SRFf = SRF.iloc[min_ws + 1:max_ws, 1:].values
        #
        # from scipy import interpolate
        # x = in_wave
        # resam_spec = np.zeros((SRFf.shape[0], spectra.shape[1]))
        # for i in range(0, spectra.shape[1]):
        #     xnew = SRF.iloc[min_ws + 1:max_ws, 0].values
        #     y = spectra[:, i]
        #     f = interpolate.interp1d(x, y, fill_value=0)
        #     resam_spec[:, i] = f(xnew)  # use interpolation function returned by `interp1d`
        #
        # spectra1 = resam_spec.T
        # indx = spectra1.shape
        # sent_asd = np.zeros((indx[0], n_bands))
        # for j in range(0, indx[0]):
        #     for i in range(0, n_bands):
        #         sent_asd[j, i] = np.sum(SRFf[:, i] * spectra1[j, :]) / np.sum(SRFf[:, i])
        # self.widget_feature_5.ax.plot(cw,sent_asd.T)
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # self.widget_feature_5.ax.set_ylabel('Value')
        # self.widget_feature_5.canvas.draw()
        #
        # df1 = pd.DataFrame(sent_asd.T, index=cw)
        # df1.to_csv(self.outputFilename + '_resample_landsat8' + '.csv', header=True, index=True)

        suffix ='_resample_landsat8' + '.csv'
        self.computeResample1(filepath,"Sheet1","Sheet0", suffix)

    def computeResample2(self, filepath, suffix):

        hdr = Utils.read_hdr_file(filepath, keep_case=False)
        fw1 = str.split(hdr['fwhm'], ',')
        fw = []
        for i in range(0, len(fw1)):
            c = str.lstrip(fw1[i])
            c = float(c)
            c = float("{0:.2f}".format(c))
            fw = np.append(fw, c)
        cw1 = str.split(hdr['wavelength'], ',')
        cw = []
        for i in range(0, len(cw1)):
            c = str.lstrip(cw1[i])
            c = float(c)
            c = float("{0:.3f}".format(c))
            cw = np.append(cw, c)

        if cw[0]<1:
            cw=[value * 1000 for value in cw]

        spectra, in_wave = self.prepare_data()
        ''' Resampling '''
        resampler = Utils.BandResampler(in_wave, cw, fwhm1=None, fwhm2=fw)
        result = np.zeros((len(cw), spectra.shape[1]))
        for i in range(0, spectra.shape[1]):
            result[:, i] = resampler(spectra[:, i])
        self.widget_feature_5.ax.plot(cw, result)
        self.widget_feature_5.ax.set_xlabel('Wavelength')
        self.widget_feature_5.ax.set_ylabel('Value')
        self.widget_feature_5.canvas.draw()

        df1 = pd.DataFrame(result, index=cw)
        df1.to_csv(self.outputFilename + suffix, header=True, index=True)

    def prisma_resample(self):
        self.widget_feature_5.ax.clear()
        filepath = os.path.join(pluginPath, 'external', 'prisma_cw_fwhm.hdr')
        # hdr = read_hdr_file(filepath, keep_case=False)
        # fw1 = str.split(hdr['fwhm'], ',')
        # fw = []
        # for i in range(0, len(fw1)):
        #     c = str.lstrip(fw1[i])
        #     c = float(c)
        #     c = float("{0:.2f}".format(c))
        #     fw = np.append(fw, c)
        # cw1 = str.split(hdr['wavelength'], ',')
        # cw = []
        # for i in range(0, len(cw1)):
        #     c = str.lstrip(cw1[i])
        #     c = float(c)
        #     c = float("{0:.3f}".format(c))
        #     cw = np.append(cw, c)
        # spectra, in_wave = self.prepare_data()
        # ''' Resampling '''
        # resampler = BandResampler(in_wave, cw, fwhm1=None, fwhm2=fw)
        # result = np.zeros((len(cw), spectra.shape[1]))
        # for i in range(0, spectra.shape[1]):
        #     result[:, i] = resampler(spectra[:, i])
        # self.widget_feature_5.ax.plot(cw,result)
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # self.widget_feature_5.ax.set_ylabel('Value')
        # self.widget_feature_5.canvas.draw()
        #
        # df1 = pd.DataFrame(result, index=cw)
        # df1.to_csv(self.outputFilename + '_resample_prisma' + '.csv', header=True, index=True)

        suffix ='_resample_prisma' + '.csv'
        self.computeResample2(filepath, suffix)

    def aviris_resample(self):
        self.widget_feature_5.ax.clear()
        filepath = os.path.join(pluginPath, 'external', 'aviris_cw_fwhm.hdr')
        # hdr = read_hdr_file(filepath, keep_case=False)
        # fw1 = str.split(hdr['fwhm'], ',')
        # fw = []
        # for i in range(0, len(fw1)):
        #     c = str.lstrip(fw1[i])
        #     c = float(c) * 1000
        #     c = float("{0:.2f}".format(c))
        #     fw = np.append(fw, c)
        # cw1 = str.split(hdr['wavelength'], ',')
        # cw = []
        # for i in range(0, len(cw1)):
        #     c = str.lstrip(cw1[i])
        #     c = float(c) * 1000
        #     c = float("{0:.3f}".format(c))
        #     cw = np.append(cw, c)
        # spectra, in_wave = self.prepare_data()
        # ''' Resampling '''
        # resampler = BandResampler(in_wave, cw, fwhm1=None, fwhm2=fw)
        # result = np.zeros((len(cw), spectra.shape[1]))
        # for i in range(0, spectra.shape[1]):
        #     result[:, i] = resampler(spectra[:, i])
        # self.widget_feature_5.ax.plot(cw,result)
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # self.widget_feature_5.ax.set_ylabel('Value')
        # self.widget_feature_5.canvas.draw()
        #
        #
        # df1 = pd.DataFrame(result, index=cw)
        # df1.to_csv(self.outputFilename + '_resample_aviris' + '.csv', header=True, index=True)

        suffix = '_resample_aviris' + '.csv'
        self.computeResample2(filepath, suffix)



    def others_resample(self):
        self.widget_feature_5.ax.clear()
        # central_wavelength_fwhm = pd.read_csv(self.Central_wavelength_filepath)
        # cw = central_wavelength_fwhm.values[:, 0]
        # fwhm = central_wavelength_fwhm.values[:, 1]

        # # HDR file read
        # hdr = read_hdr_file(self.Central_wavelength_filepath, keep_case=False)
        # fw1 = str.split(hdr['fwhm'], ',')
        # fw = []
        # for i in range(0, len(fw1)):
        #     c = str.lstrip(fw1[i])
        #     c = float(c)
        #     c = float("{0:.2f}".format(c))
        #     fw = np.append(fw, c)
        # # hdr=read_hdr_file(r'F:/SOIL HYPERSPECTRAL/chilika/roi/ang20151226t043231_corr_v2m2_img.hdr',keep_case=False)
        # cw1 = str.split(hdr['wavelength'], ',')
        # cw = []
        # for i in range(0, len(cw1)):
        #     c = str.lstrip(cw1[i])
        #     c = float(c)
        #     c = float("{0:.3f}".format(c))
        #     cw = np.append(cw, c)
        #
        # if float(cw[0])<100.0:
        #     cw=cw*1000
        #     fw=fw*1000
        #
        # spectra, in_wave = self.prepare_data()
        #
        # ''' Resampling '''
        # resampler = BandResampler(in_wave, cw, fwhm1=None, fwhm2=fw)
        # result = np.zeros((len(cw), spectra.shape[1]))
        # for i in range(0, spectra.shape[1]):
        #     result[:, i] = resampler(spectra[:, i])
        # self.widget_feature_5.ax.plot(cw,result)
        # self.widget_feature_5.ax.set_xlabel('Wavelength')
        # self.widget_feature_5.ax.set_ylabel('Value')
        # self.widget_feature_5.canvas.draw()
        #
        # df1 = pd.DataFrame(result, index=cw)
        # df1.to_csv(self.outputFilename + '_resample_others' + '.csv', header=True, index=True)

        suffix = '_resample_others' + '.csv'
        self.computeResample2(self.Central_wavelength_filepath, suffix)

    def run(self):
        self.widget_feature_5.ax.clear()
        try:

            if not (Utils.validataInputPath(self.lineEdit_3, self.Form) and \
                    Utils.validataOutputPath(self.lineEdit_4, self.Form) and \
                    Utils.validateExtension(self.lineEdit_3, 'csv', self.Form) and \
                    Utils.validateExtension(self.lineEdit_4, 'csv', self.Form)):
                return
            if self.radioButton_6.isChecked() :
                if not Utils.validateEmpty(self.lineEdit, self.Form):
                    return

            self.filepath = self.lineEdit_3.text()
            self.Central_wavelength_filepath = None
            self.outputFilename = self.lineEdit_4.text()

            print("In: " + self.filepath)
            print("Out: " + self.outputFilename)
            print("Running...")

            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            if self.radioButton.isChecked():
                self.Gisat_Hx_resample()
            if self.radioButton_7.isChecked():
                self.Gisat_Mx_resample()
            if self.radioButton_3.isChecked():
                self.prisma_resample()
            if self.radioButton_4.isChecked():
                self.aviris_resample()
            if self.radioButton_5.isChecked():
                self.sentinel2_resample()
            if self.radioButton_2.isChecked():
                self.landsat_8_resample()
            if self.radioButton_6.isChecked():


                self.Central_wavelength_filepath = self.lineEdit.text()
                self.others_resample()
            QApplication.restoreOverrideCursor()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            print(e)

