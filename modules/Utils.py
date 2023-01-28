# -*- coding: utf-8 -*-
from PyQt5.QtCore import QObject, QSettings, QFileInfo, QDir, QCoreApplication,pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5 import QtWidgets
import os

from os import path
pluginPath = os.path.split(os.path.dirname(__file__))[0]

def getLastUsedDir():
    settings = QSettings('Proximal.ini',QSettings.IniFormat)
    return settings.value("/Proximal/lastUsedDir")

# Stores last used dir in persistent settings


def setLastUsedDir(filePath):
    settings = QSettings('Proximal.ini',QSettings.IniFormat)
    # print(QSettings.fileName(settings))
    fileInfo = QFileInfo(filePath)
    if fileInfo.isDir():
        dirPath = fileInfo.filePath()
    else:
        dirPath = fileInfo.path()
    settings.setValue("/Proximal/lastUsedDir", dirPath)


def getLastSavedDir():
    settings = QSettings('Proximal.ini',QSettings.IniFormat)
    return settings.value("/Proximal/lastSavedDir")

# Stores last used dir in persistent settings


def setLastSavedDir(filePath):
    settings = QSettings('Proximal.ini',QSettings.IniFormat)
    # print(QSettings.fileName(settings))
    fileInfo = QFileInfo(filePath)
    if fileInfo.isDir():
        dirPath = fileInfo.filePath()
    else:
        dirPath = fileInfo.path()
    settings.setValue("/Proximal/lastSavedDir", dirPath)


def browseInputFile(POSTFIX,lineEditInput, filter, lineEditOutput):

    lastDataDir =getLastUsedDir()

    lineEditInput.setText("")
    fname, _ = QFileDialog.getOpenFileName(None, filter=filter, directory=lastDataDir)

    if not fname:
        lineEditInput.setText("")
        return

    if fname:
        lineEditInput.setText(fname)
        setLastUsedDir(os.path.dirname(fname))

        outputFilename = (os.path.dirname(fname)) + "/Output" + POSTFIX
        lineEditOutput.setText(outputFilename)

    return fname


def browseMetadataFile(lineEditInput, filter):

    lastDataDir = getLastUsedDir()
    lineEditInput.setText("")
    fname, _ = QFileDialog.getOpenFileName(None, filter=filter, directory=lastDataDir)
    if not fname:
        lineEditInput.setText("")
        return
    if fname:
        lineEditInput.setText(fname)
        setLastUsedDir(os.path.dirname(fname))


    return fname


def browseSaveFile(lineEditSave, filter):
    lastDataDir = getLastSavedDir()
    outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', lastDataDir, filter)
    if outputFilename:
        lineEditSave.setText(outputFilename)

    Utils.setLastSavedDir(os.path.dirname(outputFilename))

    return outputFilename


def validataInputPath(lineEditInput, parent=None):
    if (lineEditInput.text() is None) or (lineEditInput.text() == ""):
        lineEditInput.setFocus()
        QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Input File is required",
                                      QtWidgets.QMessageBox.Ok)
        return False

    if (not os.path.exists(lineEditInput.text())):
        lineEditInput.setFocus()
        QtWidgets.QMessageBox.critical(parent, "Information missing or invalid", "Kindly enter a valid input file.",
                                       QtWidgets.QMessageBox.Ok)
        return False
    return True


def validataMultpleInputPath(lineEditInput, parent=None):
    if (lineEditInput.text() is None) or (lineEditInput.text() == ""):
        lineEditInput.setFocus()
        QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Input File is required",
                                      QtWidgets.QMessageBox.Ok)
        return False

    filenames=lineEditInput.text().split(";")
    for filename in filenames:
        if (not os.path.exists(filename)):
            lineEditInput.setFocus()
            QtWidgets.QMessageBox.critical(parent, "Information missing or invalid", "Kindly enter a valid input file.",
                                           QtWidgets.QMessageBox.Ok)
            return False

    return True

def validataOutputPath(lineEditOutput, parent=None):
    if (lineEditOutput.text() is None) or (lineEditOutput.text() == ""):
        lineEditOutput.setFocus()
        QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Output File is required",
                                      QtWidgets.QMessageBox.Ok)
        return False

    if (not os.path.isdir(os.path.dirname(lineEditOutput.text()))):
        lineEditOutput.setFocus()
        QtWidgets.QMessageBox.critical(parent, "Information missing or invalid",
                                       "Kindly enter a valid output path.",
                                       QtWidgets.QMessageBox.Ok)
        return False
    return True

def validateCombobox(combobox,parent=None):
    if combobox.currentIndex() == 0:
        combobox.setFocus()
        QtWidgets.QMessageBox.information(parent, 'Message', 'Please select option first', QtWidgets.QMessageBox.Ok)
        return False
    return True

def validateEmpty(lineEdit, parent=None):
    if (lineEdit.text() is None) or (lineEdit.text() == ""):
        lineEdit.setFocus()
        QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Field cannot be left empty",
                                      QtWidgets.QMessageBox.Ok)
        return False
    return True


def validateDatatype(lineEdit, data_type, parent=None):
    try:
        number=int(lineEdit.text())
        if type(number) is not data_type:
            lineEdit.setFocus()
            QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Invalid numbers entered",
                                          QtWidgets.QMessageBox.Ok)
            return False
    except ValueError as e:
        lineEdit.setFocus()
        QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Invalid numbers entered",
                                      QtWidgets.QMessageBox.Ok)
        return False
    return True


def validateRange(lineEdit, minimum=None, maximum=None, parent=None):
    number = int(lineEdit.text())
    if minimum is not None:
        if number<minimum:
            lineEdit.setFocus()
            QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Invalid Input, number cannot < "+str(minimum),
                                          QtWidgets.QMessageBox.Ok)
            return False
    if maximum is not None:
        if number>maximum:
            lineEdit.setFocus()
            QtWidgets.QMessageBox.warning(parent, 'Information missing or invalid', "Invalid Input, number cannot > "+str(maximum),
                                          QtWidgets.QMessageBox.Ok)
            return False

    return True

def validateExtension(lineEdit,fileExt, parent=None):
    filepath = str(lineEdit.text())
    ext=os.path.basename(filepath).split('.')[-1]
    # print(ext, fileExt)
    if not str(ext) == fileExt:
        lineEdit.setFocus()
        messageDisplay = "Input file extension cannot be " + ext
        QtWidgets.QMessageBox.information(parent, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
        return False
    return True

def validateExtensionMultiple(lineEdit,fileExt, parent=None):
    filepath = str(lineEdit.text()).split(";")
    for filename in filepath:
        ext=os.path.basename(filename).split('.')[-1]
        # print(ext, fileExt)
        if not str(ext) == fileExt:
            lineEdit.setFocus()
            messageDisplay = "Input file extension cannot be " + ext
            QtWidgets.QMessageBox.information(parent, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
            return False
    return True


def validateListWidget(listWidget, parent=None):
    listVal = listWidget.currentItem()
    if (listVal is None) or (listVal == ""):
        listWidget.setFocus()
        messageDisplay = "Cannot leave Input empty!"
        QtWidgets.QMessageBox.information(parent, 'Error', messageDisplay, QtWidgets.QMessageBox.Ok)
        return False
    return True


'''
This section is for spectral indices
'''
from specdal.containers.spectrum import Spectrum
from specdal.containers.collection import Collection
import pandas as pd
import numpy as np
import math


def readGeneric( filename):
    data = pd.read_csv(filename, index_col=0)
    data_csv = data.T
    return data_csv

def readData(filename):
    s = Spectrum(filepath=filename)
    df = s.measurement
    return df

def readSVC(filename):
    df = pd.read_csv(filename, error_bad_lines=False, warn_bad_lines=False)
    # s = Spectrum(filepath=filename)
    # # print(s.measurement)
    # df = s.measurement
    # print(df)
    # return df
    if len(df) == 516:

        data = pd.DataFrame(df)
        data.rename(columns={'///GER SIGNATUR FILE///': 'col2'}, inplace=True)
        # Drop rows
        data.drop(data.head(4).index, inplace=True)
        # data.rename(columns={'0': 'col2'}, inplace=True)
        data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
        reference = data.col2.str.split(' ', expand=True)[2]
        target = data.col2.str.split(' ', expand=True)[4]
        # data['wavelength'] = data['wavelength'].astype('int64')
        data.wavelength = data.wavelength.astype('float')
        reference = reference.astype('float')
        target = target.astype('float')

        # data.reset_index(drop=True, inplace=True)
        # data.index.name = 'Index'
        data['Reflectance'] = target / reference
        data.drop(['col2'], axis=1, inplace=True)
        data.index = data.wavelength
        # Remove column name
        data.drop(['wavelength'], axis=1, inplace=True)
        return data

        # print(data)
    else:
        data = pd.DataFrame(df)
        data.rename(columns={'/*** Spectra Vista SIG Data ***/': 'col2'}, inplace=True)
        # Drop rows
        data.drop(data.head(5).index, inplace=True)
        # data.rename(columns={'0': 'col2'}, inplace=True)
        # print(data)
        data['wavelength'] = data.col2.str.split(' ', expand=True)[0]
        reference = data.col2.str.split(' ', expand=True)[2]
        target = data.col2.str.split(' ', expand=True)[4]
        Reflectance = data.col2.str.split(' ', expand=True)[6]
        # print(data)
        # data['wavelength'] = data['wavelength'].astype('float')
        # data['reference'] = data['reference'].astype('float')
        # data['target'] = data['target'].astype('float')
        Reflectance = Reflectance.astype('float')
        data.wavelength = data.wavelength.astype('float')
        data['Reflectance'] = Reflectance / 100
        data.drop(['col2'], axis=1, inplace=True)
        data.index = data.wavelength

        # Remove column name
        data.drop(['wavelength'], axis=1, inplace=True)

        return data

    def getdataframe(self, filename):
        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df2.index.name = 'wavelength'
        for i in filename:
            df1 = df.append(self.readASD(i)).transpose()
            df1.index.name = 'wavelength'
            df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)
        return df2

def calculateSpectralIndices(listVal,filenames,spectro_type):
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df2.index.name = 'wavelength'
    filename_cols=[]
    for filename in filenames:
        if spectro_type=='ASD':
            df1 = df.append(readData(filename)).transpose()
        elif spectro_type=='SVC':
            df1 = df.append(readData(filename)).transpose()
        else:
            df1 = df.append(readGeneric(filename)).transpose()

        filename_cols.append(os.path.basename(filename))
        df1.index.name = 'wavelength'
        df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)

    arr = np.array(df2.index)
    filepath = os.path.join(pluginPath, 'external', 'Vegetation_index.csv')
    csv_read = pd.read_csv(filepath)

    name_ind = ["ARI", "ARI2", "BG1", "BG2", "BRI1", "BRI2", "CAI", "CRI550", "CRI700", "DI1", "GM1", "GM2",
                "Greenness index(G)", "LIC3", "HI", "MCARI", "MRENDVI", "MRESR", "MSAVI", "MSR", "MSI", "MTVI",
                "NDI1", "NDI2", "NDI3", "NDNI", "NDLI", "NDVI", "NMDI", "OSAVI", "PRI", "PSRI", "PSNDc", "PSSRa",
                "PSSRb", "PSSRc", "RARS", "RDVI", "RENDVI", "SIPI", "SR", "SR2", "SR3", "SR4", "SR5", "SR6", "SR7",
                "TCARI", "TGI", "VREI1", "VREI2", "WBI", "ZM", "LMVI1", "RVI"]

    if (listVal == 'ALL'):
        index_val = []
        for i in range(0, 55):
            stri = csv_read.iloc[i, 0]
            if (stri == name_ind[i]):
                check1 = csv_read.loc[i].iloc[1]
                check2 = csv_read.loc[i].iloc[2]
                check3 = csv_read.loc[i].iloc[3]
                check4 = csv_read.loc[i].iloc[4]
                check5 = csv_read.loc[i].iloc[5]

                difference_array_V1 = np.absolute(arr - check1)
                # print(difference_array_V1, 'diff')
                # find the index of minimum element from the array
                index = difference_array_V1.argmin()
                # print(index,'index')
                # print("Nearest element to the given values is : ", arr[index])
                difference_array_V2 = np.absolute(arr - check2)
                # print(difference_array_V2, 'diff')
                # find the index of minimum element from the array
                index2 = difference_array_V2.argmin()
                # print(index2, 'index')
                # print("Nearest element to the given values is : ", arr[index2])
                difference_array_V3 = np.absolute(arr - check3)
                # print(difference_array_V3, 'diff')
                # find the index of minimum element from the array
                index3 = difference_array_V3.argmin()
                # print("Nearest element to the given values is : ", arr[index3])
                difference_array_V4 = np.absolute(arr - check4)
                # print(difference_array_V4, 'diff')
                # find the index of minimum element from the array
                index4 = difference_array_V4.argmin()
                # print("Nearest element to the given values is : ", arr[index4])
                difference_array_V5 = np.absolute(arr - check5)
                # print(difference_array_V5, 'diff')
                # find the index of minimum element from the array
                index5 = difference_array_V5.argmin()
                # print("Nearest element to the given values is : ", arr[index5])
                V1 = arr[index]
                V2 = arr[index2]
                V3 = arr[index3]
                V4 = arr[index4]
                V5 = arr[index5]
                C1 = csv_read.loc[i].iloc[6]
                C2 = csv_read.loc[i].iloc[7]
                C3 = csv_read.loc[i].iloc[8]
                C4 = csv_read.loc[i].iloc[9]
                if (np.isnan(V1)):
                    V1 = 0
                if (np.isnan(V2)):
                    V2 = 0
                if (np.isnan(V3)):
                    V3 = 0
                if (np.isnan(V4)):
                    V4 = 0
                if (np.isnan(V5)):
                    V5 = 0
                if (np.isnan(C1)):
                    C1 = 0
                if (np.isnan(C2)):
                    C2 = 0
                if (np.isnan(C3)):
                    C3 = 0
                if (np.isnan(C4)):
                    C4 = 0
                val = getIndices(df2, stri, V1, V2, V3, V4, V5, C1, C2, C3, C4)
            index_val.append(val)
            # print(index_val,str(i))

        all_indices_df = pd.DataFrame(index_val, index=name_ind)
        return all_indices_df

    else:
        val=0.0
        result={}
        for i in range(0, 55):
            stri = csv_read.iloc[i, 0]
            if (stri == listVal):
                # V1 = index.loc[i].iloc[1]
                # V2 = index.loc[i].iloc[2]
                # V3 = index.loc[i].iloc[3]
                # V4 = index.loc[i].iloc[4]
                # V5 = index.loc[i].iloc[5]
                check1 = csv_read.loc[i].iloc[1]
                check2 = csv_read.loc[i].iloc[2]
                check3 = csv_read.loc[i].iloc[3]
                check4 = csv_read.loc[i].iloc[4]
                check5 = csv_read.loc[i].iloc[5]

                difference_array_V1 = np.absolute(arr - check1)
                # print(difference_array_V1, 'diff')
                # find the index of minimum element from the array
                index = difference_array_V1.argmin()
                # print(index,'index')
                # print("Nearest element to the given values is : ", arr[index])
                difference_array_V2 = np.absolute(arr - check2)
                # print(difference_array_V2, 'diff')
                # find the index of minimum element from the array
                index2 = difference_array_V2.argmin()
                # print(index2, 'index')
                # print("Nearest element to the given values is : ", arr[index2])
                difference_array_V3 = np.absolute(arr - check3)
                # print(difference_array_V3, 'diff')
                # find the index of minimum element from the array
                index3 = difference_array_V3.argmin()
                # print("Nearest element to the given values is : ", arr[index3])
                difference_array_V4 = np.absolute(arr - check4)
                # print(difference_array_V4, 'diff')
                # find the index of minimum element from the array
                index4 = difference_array_V4.argmin()
                # print("Nearest element to the given values is : ", arr[index4])
                difference_array_V5 = np.absolute(arr - check5)
                # print(difference_array_V5, 'diff')
                # find the index of minimum element from the array
                index5 = difference_array_V5.argmin()
                # print("Nearest element to the given values is : ", arr[index5])
                V1 = arr[index]
                V2 = arr[index2]
                V3 = arr[index3]
                V4 = arr[index4]
                V5 = arr[index5]
                C1 = csv_read.loc[i].iloc[6]
                C2 = csv_read.loc[i].iloc[7]
                C3 = csv_read.loc[i].iloc[8]
                C4 = csv_read.loc[i].iloc[9]
                if (np.isnan(V1)):
                    V1 = 0
                if (np.isnan(V2)):
                    V2 = 0
                if (np.isnan(V3)):
                    V3 = 0
                if (np.isnan(V4)):
                    V4 = 0
                if (np.isnan(V5)):
                    V5 = 0
                if (np.isnan(C1)):
                    C1 = 0
                if (np.isnan(C2)):
                    C2 = 0
                if (np.isnan(C3)):
                    C3 = 0
                if (np.isnan(C4)):
                    C4 = 0
                val = getIndices(df2, stri, V1, V2, V3, V4, V5, C1, C2, C3, C4)
                print(str(listVal), val.to_numpy())

                # self.vegindex = val
                # self.lineEditOutput.setText(self.filename)
        val=val.to_numpy()
        for i,c in enumerate(filename_cols):
            result[c]=val[i]

        # single_indices_df = pd.DataFrame(list(val.to_numpy()), index=[str(listVal)], columns=[filename_cols])
        single_indices_df = pd.DataFrame(result, index=[str(listVal)])
        return single_indices_df

def getIndices(df, str1, a1, a2, a3, a4, a5, c1, c2, c3, c4):
    if (a1 != 0):
        v1 = df.loc[a1]
    if (a2 != 0):
        v2 = df.loc[a2]
    if (a3 != 0):
        v3 = df.loc[a3]
    if (a4 != 0):
        v4 = df.loc[a4]
    # if(a5!=0):
    # v5=df.loc[a5]
    if str1 == "ARI":
        return (1 / v1) - (1 / v2)

    if str1 == "ARI2":
        return v1 * ((1 / v2) - (1 / v3))
    if str1 == "BG1":
        return v1 / v2
    if str1 == "BG2":
        return v1 / v2
    if str1 == "BRI1":
        return v1 / v2
    if str1 == "BRI2":
        return v1 / v2
    if str1 == "CAI":
        return c1 * (v1 + v2) - v3
    if str1 == "CRI550":
        return (1 / v1) / (1 / v2)
    if str1 == "CRI700":
        return (1 / v1) / (1 / v2)
    if str1 == "DI1":
        return v1 - v2
    if str1 == "GM1":
        return v1 / v2
    if str1 == "GM2":
        return v1 / v2
    if str1 == "Greenness index(G)":
        return v1 / v2
    if str1 == "LIC3":
        return v1 / v2
    if str1 == "HI":
        return ((v1 - v2) / (v1 + v2)) - c1 * v3
    if str1 == "MCARI":
        return ((v1 - v2) - c1 * (v1 - v3)) * (v1 / v2)
    if str1 == "MRENDVI":
        return (v1 - v2) / (v1 + v2 - c1 * v3)
    if str1 == "MRESR":
        return (v1 - v2) / (v1 + v2)
    if str1 == "MSAVI":
        return c1 * (c2 * v1 + 1 - np.sqrt(((c2 * v1 + 1) ** 2) - c3 * (v1 - v2)))

    if str1 == "MSR":
        return ((v1 / v2) - 1) / (np.sqrt(v1 / v2 + 1))
    if str1 == "MSI":
        return v1 / v2
    if str1 == "MTVI":
        return c1 * (c1 * (v1 - v2) - c2 * (v3 - v2))
    if str1 == "NDI1":
        return (v1 - v2) / (v1 - v3)
    if str1 == "NDI2":
        return (v1 - v2) / (v1 - v3)
    if str1 == "NDI3":
        return (v1 - v2) / (v3 - v4)

    if str1 == "NDNI":
        return (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))
    if str1 == "NDLI":
        return (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))

    if str1 == "NDVI":
        return (v1 - v2) / (v1 + v2)
    if str1 == "NMDI":
        return (v1 - (v2 - v3)) / (v1 + (v2 - v3))
    if str1 == "OSAVI":
        return ((1 + c1) * (v1 - v2)) / (v1 + v2 + c2)
    if str1 == "PRI":
        return (v1 - v2) / (v1 + v2)
    if str1 == "PSRI":
        return (v1 - v2) / v3
    if str1 == "PSNDc":
        return (v1 - v2) / (v1 + v2)
    if str1 == "PSSRa":
        return v1 / v2
    if str1 == "PSSRb":
        return v1 / v2
    if str1 == "PSSRc":
        return v1 / v2
    if str1 == "RARS":
        return v1 / v2
    if str1 == "RDVI":
        return (v1 - v2) / (np.sqrt(v1 + v2))
    if str1 == "RENDVI":
        return (v1 - v2) / (v1 + v2)
    if str1 == "SIPI":
        return (v1 - v2) / (v1 - v3)
    if str1 == "SR":
        return v1 / v2
    if str1 == "SR2":
        return v1 / v2
    if str1 == "SR3":
        return v1 / v2
    if str1 == "SR4":
        return v1 / v2
    if str1 == "SR5":
        return v1 / (v2 * v3)
    if str1 == "SR6":
        return v1 / (v2 * v3)
    if str1 == "SR7":
        return v1 / (v2 * v3)
    if str1 == "TCARI":
        return c1 * ((v1 - v2) - c2 * (v1 - v3) * (v1 / v2))
    if str1 == "VREI1":
        return v1 / v2
    if str1 == "VREI2":
        return (v1 - v2) / (v3 - v4)
    if str1 == "WBI":
        return v1 / v2
    if str1 == "TGI":
        return c1 * ((v1 - v2) * (v3 - v4) - (v1 - v4) * (v1 - v2))
    if str1 == "ZM":
        return v1 / v2
    if str1 == "LMVI1":
        return (v1 - v2) / (v1 + v2)
    if str1 == "RVI":
        return v1 / v2
    if str1 == 'ALL':
        # ari=v1 / v2
        # rvi=v1 / v2
        # return {"ARI":ari,"RVI":rvi}
        ARI = (1 / v1) - (1 / v2)
        ARI2 = v1 * ((1 / v2) - (1 / v3))
        BG1 = v1 / v2
        BG2 = v1 / v2
        BRI1 = v1 / v2
        BRI2 = v1 / v2
        CAI = c1 * (v1 + v2) - v3
        CRI550 = (1 / v1) / (1 / v2)
        CRI700 = (1 / v1) / (1 / v2)
        DI1 = v1 - v2
        GM1 = v1 / v2
        GM2 = v1 / v2
        Greenness_index = v1 / v2
        LIC3 = v1 / v2
        HI = ((v1 - v2) / (v1 + v2)) - c1 * v3
        MCARI = ((v1 - v2) - c1 * (v1 - v3)) * (v1 / v2)
        MRENDVI = (v1 - v2) / (v1 + v2 - c1 * v3)
        MRESR = (v1 - v2) / (v1 + v2)
        MSAVI = c1 * (c2 * v1 + 1 - np.sqrt(((c2 * v1 + 1) ** 2) - c3 * (v1 - v2)))
        MSR = ((v1 / v2) - 1) / (np.sqrt(v1 / v2 + 1))
        MSI = v1 / v2
        MTVI = c1 * (c1 * (v1 - v2) - c2 * (v3 - v2))
        NDI1 = (v1 - v2) / (v1 - v3)
        NDI2 = (v1 - v2) / (v1 - v3)
        NDI3 = (v1 - v2) / (v3 - v4)
        NDNI = (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))
        NDLI = (np.log(1 / v1) - np.log(1 / v2)) / (np.log(1 / v1) + np.log(1 / v2))
        NDVI = (v1 - v2) / (v1 + v2)
        NMDI = (v1 - (v2 - v3)) / (v1 + (v2 - v3))
        OSAVI = ((1 + c1) * (v1 - v2)) / (v1 + v2 + c2)
        PRI = (v1 - v2) / (v1 + v2)
        PSRI = (v1 - v2) / v3
        PSNDc = (v1 - v2) / (v1 + v2)
        PSSRa = v1 / v2
        PSSRb = v1 / v2
        PSSRc = v1 / v2
        RARS = v1 / v2
        RDVI = (v1 - v2) / (np.sqrt(v1 + v2))
        RENDVI = (v1 - v2) / (v1 + v2)
        SIPI = (v1 - v2) / (v1 - v3)
        SR = v1 / v2
        SR2 = v1 / v2
        SR3 = v1 / v2
        SR4 = v1 / v2
        SR5 = v1 / (v2 * v3)
        SR6 = v1 / (v2 * v3)
        SR7 = v1 / (v2 * v3)
        TCARI = c1 * ((v1 - v2) - c2 * (v1 - v3) * (v1 / v2))
        VREI1 = v1 / v2
        VREI2 = (v1 - v2) / (v3 - v4)
        WBI = v1 / v2
        TGI = c1 * ((v1 - v2) * (v3 - v4) - (v1 - v4) * (v1 - v2))
        ZM = v1 / v2
        LMVI1 = (v1 - v2) / (v1 + v2)
        RVI = v1 / v2
        all_ind = [ARI, ARI2, BG1, BG2, BRI1, BRI2, CAI, CRI550, CRI700, DI1, GM1, GM2, Greenness_index, LIC3, HI,
                   MCARI, MRENDVI, MRESR, MSAVI, MSR, MSI, MTVI, NDI1, NDI2, NDI3, NDNI, NDLI, NDVI, NMDI, OSAVI, PRI,
                   PSRI, PSNDc, PSSRa, PSSRb, PSSRc, RARS, RDVI, RENDVI, SIPI, SR, SR2, SR3, SR4, SR5, SR6, SR7, TCARI,
                   TGI, VREI1, VREI2, WBI, ZM, LMVI1, RVI]
        name_ind = ["ARI", "ARI2", "BG1", "BG2", "BRI1", "BRI2", "CAI", "CRI550", "CRI700", "DI1", "GM1", "GM2",
                    "Greenness index(G)", "LIC3", "HI", "MCARI", "MRENDVI", "MRESR", "MSAVI", "MSR", "MSI", "MTVI",
                    "NDI1", "NDI2", "NDI3", "NDNI", "NDLI", "NDVI", "NMDI", "OSAVI", "PRI", "PSRI", "PSNDc", "PSSRa",
                    "PSSRb", "PSSRc", "RARS", "RDVI", "RENDVI", "SIPI", "SR", "SR2", "SR3", "SR4", "SR5", "SR6", "SR7",
                    "TCARI", "TGI", "VREI1", "VREI2", "WBI", "ZM", "LMVI1", "RVI"]
        # dict = {'RVI':RVI, 'ARI': ARI,"ARI2":ARI2}
        # df = pd.DataFrame({'Spectral index': name_ind, 'Spectral index value': all_ind})
        return all_ind


def run_spectral_indices(type_spectro,lineEditInput,lineEditOutput,listWidget, parent):
    if type_spectro=='ASD':
        extension='asd'
    elif type_spectro=='SVC':
        extension = 'sig'
    else:
        extension = 'csv'
    if not (validataMultpleInputPath(lineEditInput, parent) and \
            validataOutputPath(lineEditOutput, parent) and \
            validateExtensionMultiple(lineEditInput, extension, parent) and \
            validateExtension(lineEditOutput, 'csv', parent) and \
            validateListWidget(listWidget)):
        return

    inputFiles = str(lineEditInput.text()).split(";")
    outputFile = str(lineEditOutput.text())

    listVal = listWidget.currentItem().text()
    print("Selected Indices : ", listVal)
    result_df = calculateSpectralIndices(listVal, inputFiles, type_spectro)
    result_df.to_csv(outputFile, index_label='Spectral Indices')

    print("Successfully computed selected indices")



'''
Some common functions for Time Series and Visualizer
'''
def feature_select(X, y, algo):
    from sklearn.feature_selection import f_classif, mutual_info_regression
    from sklearn.feature_selection import SelectFromModel

    if algo == "classif":
        f_test, pval = f_classif(X, y)
        f_test /= np.max(f_test)
        score = -np.log10(pval)
        score /= score.max()
        return f_test, score

    if algo == "mutual":
        mi = mutual_info_regression(X, y)
        mi /= np.max(mi)
        pval = 0 * mi
        return mi, pval

    # if algo=="L1":
    #     from sklearn.svm import LinearSVC
    #     lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    #     model = SelectFromModel(lsvc, prefit=True)
    #     X_new = model.transform(X)
    #     pval=0*lsvc.feature_importances_
    #     return lsvc.feature_importances_, pval

    if algo == "Tree":
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X)
        pval = 0 * clf.feature_importances_
        return clf.feature_importances_, pval

    if algo == "SEP":
        m1 = X[np.ndarray.flatten(y == 0), :].mean(axis=0)
        m2 = X[np.ndarray.flatten(y == 1), :].mean(axis=0)
        s1 = X[np.ndarray.flatten(y == 0), :].std(axis=0)
        s2 = X[np.ndarray.flatten(y == 1), :].std(axis=0)
        SDI = (np.abs(m1 - m2)) / (s1 + s2)
        SDI /= np.max(SDI)
        return SDI, SDI * 0

def twoClassPlot(df_list,mplWidget, combobox_A,combobox_B, parent=None):
    mplWidget.clear()

    wavelength,df_mean,crop,dupl_obs_samples=df_list

    if combobox_A.currentIndex() < 1:
        QtWidgets.QMessageBox.information(parent, 'Error', 'Please select first Class A',
                                          QtWidgets.QMessageBox.Ok)
        combobox_A.setFocus()
        return

    if combobox_B.currentIndex() < 1:
        QtWidgets.QMessageBox.information(parent, 'Error', 'Please select first Class B',
                                          QtWidgets.QMessageBox.Ok)
        combobox_B.setFocus()
        return

    label_one = str(int(combobox_A.currentText().split()[1]))
    label_two = str(int(combobox_B.currentText().split()[1]))
    spectra1 = df_mean[label_one]
    spectra2 =df_mean[label_two]
    mplWidget.ax.plot(wavelength, spectra1,
                                     label='Class = ' + str(crop[dupl_obs_samples[int(label_one) - 1]]),
                                     # label='Class = ' + str(label_one),
                                     linewidth=3)
    mplWidget.ax.plot(wavelength, spectra2,
                                     label='Class = ' + str(crop[dupl_obs_samples[int(label_two) - 1]]),
                                     # label='Class = ' + str(label_two),
                                     linewidth=3)
    mplWidget.ax.set_ylabel('Reflectance')
    mplWidget.ax.set_xlabel('Wavelength')

    legend = mplWidget.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    legend.set_draggable(True)
    mplWidget.canvas.draw()


def populateWavelength(wavelengthCmb,filepath):
    df_spectra = pd.read_csv(filepath,
                             header=0, index_col=0)
    wavelength = df_spectra.index.values
    wavelengthCmb.clear()
    for i in wavelength:
        wavelengthCmb.addItem(str(i))


def getHeaderInfo(path=None, header=[]):
    if not isinstance(path, str):

        return 'Invalid path',False

    if not os.path.isfile(path):
        return 'Invalid path',False

    # mbytes = os.path.getsize(path) / 1000 ** 2
    # if mbytes > MAX_CSV_SIZE:
    #     return False
    msg = 'Invalid CSV format, data should start with Columns : '+str(header[0])
    try:

        f = open(path, 'r', encoding='utf-8')
        text = f.read()
        f.close()

        line = text.splitlines(keepends=True)[0]
        if line.split(',')[0] not in header:
            return msg,False

        return line, True
    except Exception as e:
        return msg,False

def validateInputFormat(inFile):
    msg, status_1 = getHeaderInfo(inFile, ['Wavelength', 'wavelength'])
    if not status_1:
        QtWidgets.QMessageBox.information(None, 'Error', msg,
                                          QtWidgets.QMessageBox.Ok)
    return status_1

def validateMetaFormat(inFile, fname):
    header_line, status_1 = getHeaderInfo(inFile, ['Wavelength', 'wavelength'])
    if status_1:
        prop_line, status_2 = getHeaderInfo(fname, ['Spectra ID', 'spectra id'])
        if not status_2:
            QtWidgets.QMessageBox.information(None, 'Error', prop_line,
                                              QtWidgets.QMessageBox.Ok)
            return False

        if status_2 and (len(prop_line) != len(header_line)):
            messageDisplay = "Non Matching Property file with input spectral file "
            QtWidgets.QMessageBox.information(None, 'Error', messageDisplay,
                                              QtWidgets.QMessageBox.Ok)
            return False
    return True





def erf(z):
    '''The error function (used to calculate the gaussian integral).'''
    import math
    t = 1.0 / (1.0 + 0.5 * abs(z))
    # use Horner's method
    ans = 1 - t * math.exp(-z * z - 1.26551223 +
                           t * (1.00002368 +
                                t * (0.37409196 +
                                     t * (0.09678418 +
                                          t * (-0.18628806 +
                                               t * (0.27886807 +
                                                    t * (-1.13520398 +
                                                         t * (1.48851587 +
                                                              t * (-0.82215223 +
                                                                   t * (0.17087277))))))))))
    if z >= 0.0:
        return ans
    else:
        return -ans


def erfc(z):
    '''Complement of the error function.'''
    return 1.0 - erf(z)


def normal_cdf(x):
    '''CDF of the normal distribution.'''
    sqrt2 = 1.4142135623730951
    return 0.5 * erfc(-x / sqrt2)


def normal_integral(a, b):
    '''Integral of the normal distribution from a to b.'''
    return normal_cdf(b) - normal_cdf(a)


def ranges_overlap(R1, R2):
    '''Returns True if there is overlap between ranges of pairs R1 and R2.'''
    if (R1[0] < R2[0] and R1[1] < R2[0]) or \
            (R1[0] > R2[1] and R1[1] > R2[1]):
        return False
    return True


def overlap(R1, R2):
    '''Returns (min, max) of overlap between the ranges of pairs R1 and R2.'''
    return (max(R1[0], R2[0]), min(R1[1], R2[1]))


def normal(mean, stdev, x):
    from math import exp, pi
    sqrt_2pi = 2.5066282746310002
    return exp(-((x - mean) / stdev) ** 2 / 2.0) / (sqrt_2pi * stdev)


def build_fwhm(centers):
    '''Returns FWHM list, assuming FWHM is midway between adjacent bands.
    '''
    fwhm = [0] * len(centers)
    fwhm[0] = centers[1] - centers[0]
    fwhm[-1] = centers[-1] - centers[-2]
    for i in range(1, len(centers) - 1):
        fwhm[i] = (centers[i + 1] - centers[i - 1]) / 2.0
    return fwhm


def create_resampling_matrix(centers1, fwhm1, centers2, fwhm2):
    '''
    Returns a resampling matrix to convert spectra from one band discretization
    to another.  Arguments are the band centers and full-width half maximum
    spectral response for the original and new band discretizations.
    '''
    import numpy

    sqrt_8log2 = 2.3548200450309493

    N1 = len(centers1)
    N2 = len(centers2)
    bounds1 = [[centers1[i] - fwhm1[i] / 2.0, centers1[i] + fwhm1[i] /
                2.0] for i in range(N1)]
    bounds2 = [[centers2[i] - fwhm2[i] / 2.0, centers2[i] + fwhm2[i] /
                2.0] for i in range(N2)]

    M = numpy.zeros([N2, N1])

    jStart = 0
    nan = float('nan')
    for i in range(N2):
        stdev = fwhm2[i] / sqrt_8log2
        j = jStart

        # Find the first original band that overlaps the new band
        while j < N1 and bounds1[j][1] < bounds2[i][0]:
            j += 1

        if j == N1:
            print(('No overlap for target band %d (%f / %f)' % (
                i, centers2[i], fwhm2[i])))
            M[i, 0] = nan
            continue

        matches = []

        # Get indices for all original bands that overlap the new band
        while j < N1 and bounds1[j][0] < bounds2[i][1]:
            if ranges_overlap(bounds1[j], bounds2[i]):
                matches.append(j)
            j += 1

        # Put NaN in first element of any row that doesn't produce a band in
        # the new schema.
        if len(matches) == 0:
            print(('No overlap for target band %d (%f / %f)' % (
                i, centers2[i], fwhm2[i])))
            M[i, 0] = nan
            continue

        # Determine the weights for the original bands that overlap the new
        # band. There may be multiple bands that overlap or even just a single
        # band that only partially overlaps.  Weights are normoalized so either
        # case can be handled.

        overlaps = [overlap(bounds1[k], bounds2[i]) for k in matches]
        contribs = numpy.zeros(len(matches))
        A = 0.
        for k in range(len(matches)):
            # endNorms = [normal(centers2[i], stdev, x) for x in overlaps[k]]
            # dA = (overlaps[k][1] - overlaps[k][0]) * sum(endNorms) / 2.0
            (a, b) = [(x - centers2[i]) / stdev for x in overlaps[k]]
            dA = normal_integral(a, b)
            contribs[k] = dA
            A += dA
        contribs = contribs / A
        for k in range(len(matches)):
            M[i, matches[k]] = contribs[k]
    return M


class BandResampler:
    '''A callable object for resampling spectra between band discretizations.

    A source band will contribute to any destination band where there is
    overlap between the FWHM of the two bands.  If there is an overlap, an
    integral is performed over the region of overlap assuming the source band
    data value is constant over its FWHM (since we do not know the true
    spectral load over the source band) and the destination band has a Gaussian
    response function. Any target bands that do not have any overlapping source
    bands will contain NaN as the resampled band value.

    If bandwidths are not specified for source or destination bands, the bands
    are assumed to have FWHM values that span half the distance to the adjacent
    bands.
    '''

    def __init__(self, centers1, centers2, fwhm1=None, fwhm2=None):
        '''BandResampler constructor.

        Usage:

                resampler = BandResampler(bandInfo1, bandInfo2)

                resampler = BandResampler(centers1, centers2, [fwhm1 = None [, fwhm2 = None]])

            Arguments:

                `bandInfo1` (:class:`~spectral.BandInfo`):

                    Discretization of the source bands.

                `bandInfo2` (:class:`~spectral.BandInfo`):

                    Discretization of the destination bands.

                `centers1` (list):

                    floats defining center values of source bands.

                `centers2` (list):

                    floats defining center values of destination bands.

                `fwhm1` (list):

                    Optional list defining FWHM values of source bands.

                `fwhm2` (list):

                    Optional list defining FWHM values of destination bands.

            Returns:

                A callable BandResampler object that takes a spectrum corresponding
                to the source bands and returns the spectrum resampled to the
                destination bands.

            If bandwidths are not specified, the associated bands are assumed to
            have FWHM values that span half the distance to the adjacent bands.
            '''
        #        from spectral.spectral import BandInfo
        #        if isinstance(centers1, BandInfo):
        #            fwhm1 = centers1.bandwidths
        #            centers1 = centers1.centers
        #        if isinstance(centers2, BandInfo):
        #            fwhm2 = centers2.bandwidths
        #            centers2 = centers2.centers
        if fwhm1 is None:
            fwhm1 = build_fwhm(centers1)
        if fwhm2 is None:
            fwhm2 = build_fwhm(centers2)
        self.matrix = create_resampling_matrix(
            centers1, fwhm1, centers2, fwhm2)

    def __call__(self, spectrum):
        '''Takes a source spectrum as input and returns a resampled spectrum.

            Arguments:

                `spectrum` (list or :class:`numpy.ndarray`):

                    list or vector of values to be resampled.  Must have same
                    length as the source band discretiation used to created the
                    resampler.

            Returns:

                A resampled rank-1 :class:`numpy.ndarray` with length corresponding
                to the destination band discretization used to create the resampler.

            Any target bands that do not have at lease one overlapping source band
            will contain `float('nan')` as the resampled band value.'''
        import numpy
        return numpy.dot(self.matrix, spectrum)
import re
import collections
def read_hdr_file(hdrfilename, keep_case=False):
    """
        Read information from ENVI header file to a dictionary.
        By default all keys are converted to lowercase. To stop this behaviour
        and keep the origional case set 'keep_case = True'
    """
    ENVI_TO_NUMPY_DTYPE = {'1': np.uint8,
                           '2': np.int16,
                           '3': np.int32,
                           '4': np.float32,
                           '5': np.float64,
                           '6': np.complex64,
                           '9': np.complex128,
                           '12': np.uint16,
                           '13': np.uint32,
                           '14': np.int64,
                           '15': np.uint64}
    output = collections.OrderedDict()
    comments = ""
    inblock = False
    try:
        hdrfile = open(hdrfilename, "r")
    except:
        raise IOError("Could not open hdr file " + str(hdrfilename) + \
                      ". Reason: " + str(sys.exc_info()[1]), sys.exc_info()[2])
        # Read line, split it on equals, strip whitespace from resulting strings
        # and add key/value pair to output
    for currentline in hdrfile:
        # ENVI headers accept blocks bracketed by curly braces - check for these
        if not inblock:
            # Check for a comment
            if re.search("^;", currentline) is not None:
                comments += currentline
            # Split line on first equals sign
            elif re.search("=", currentline) is not None:
                linesplit = re.split("=", currentline, 1)
                key = linesplit[0].strip()
                # Convert all values to lower case unless requested to keep.
                if not keep_case:
                    key = key.lower()
                value = linesplit[1].strip()
                # If value starts with an open brace, it's the start of a block
                # - strip the brace off and read the rest of the block
                if re.match("{", value) is not None:
                    inblock = True
                    value = re.sub("^{", "", value, 1)
                    # If value ends with a close brace it's the end
                    # of the block as well - strip the brace off
                    if re.search("}$", value):
                        inblock = False
                        value = re.sub("}$", "", value, 1)
                value = value.strip()
                output[key] = value
        else:
            # If we're in a block, just read the line, strip whitespace
            # (and any closing brace ending the block) and add the whole thing
            value = currentline.strip()
            if re.search("}$", value):
                inblock = False
                value = re.sub("}$", "", value, 1)
                value = value.strip()
            output[key] = output[key] + value
    hdrfile.close()
    output['_comments'] = comments
    return output



'''
Common code blocks for ViewData
'''
import matplotlib.pyplot as plt
from modules.PandasModel import PandasModel

def readViewData(filename, operation):
    s = Spectrum(filepath=filename, measure_type=operation)
    df = s.measurement
    return df


def getData(filepaths,operation):
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df2.index.name = 'wavelength'
    for i in filepaths:  # list of filenames
        df1 = df.append(readViewData(i, operation)).transpose()
        df1.index.name = 'wavelength'
        df2 = df2.merge(df1, how='outer', left_index=True, right_index=True)

    file_paths = []
    for i in range(0, len(filepaths)):
        file_paths.append(os.path.basename(filepaths[i]))
    # print(file_paths,"fileapth")
    df2.columns = file_paths
    df2 = df2.reset_index()

    return df2

def loadTable(filepaths,operation,tableView):
    df2=getData(filepaths,operation)
    model = PandasModel(df2.T)
    tableView.setModel(model)
    return df2,model


def plotGraph(filepaths, operation):

    df2=getData(filepaths,operation)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(0, len(filepaths)):
        label = os.path.basename(filepaths[i])
        ax.plot(df2.iloc[:, 0],df2.iloc[:, i+1], label=label)
        ax.legend()
    # xtick = list(df2.index)
    # ax.set_xticks(xtick[::100])
    plt.xlabel("wavelength")
    if operation == 'pct_reflect':
        plt.ylabel('reflectance')
    elif operation == 'tgt_reflect' or operation == 'tgt_radiance':
        plt.ylabel('radiance')
    plt.show()


def saveTable(df,tableView,outputFilename):
    model = tableView.model()
    if model is None:
        messageDisplay = "Empty data cannot be saved!"
        QtWidgets.QMessageBox.information(None, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
        return

    if (model.rowCount() == 0):
        messageDisplay = "Empty data cannot be saved!"
        QtWidgets.QMessageBox.information(None, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)
        return

    df.to_csv(outputFilename, index=False)

    messageDisplay = "Data saved sucessfully!"
    QtWidgets.QMessageBox.information(None, 'Message', messageDisplay, QtWidgets.QMessageBox.Ok)


def browseMultipleFiles(POSTFIX ,filters,lineEditInput,lineEditOutput):
    lastDataDir = getLastUsedDir()
    fnames, _ = QFileDialog.getOpenFileNames(None, filter=filters, directory=lastDataDir)
    if not fnames:
        lineEditOutput.setText("")
        return

    if fnames:
        if len(fnames)>1:
            inputText = ';'.join(fnames)
        else:
            inputText=fnames[0]

        lineEditInput.setText(inputText)
        setLastUsedDir(os.path.dirname(fnames[0]))
        filename = os.path.dirname(fnames[0]) + "/Output" + POSTFIX
        lineEditOutput.setText(filename)
    return fnames


def getOperation(svcReflRb,svcRadRb, svcWhiteRb, spec):
    if (svcReflRb.isChecked() == True):
        operation = "pct_reflect"
    elif (svcRadRb.isChecked() == True):
        if spec=='SEV':
            operation = "tgt_reflect"
        else:
            operation = "tgt_radiance"
    elif (svcWhiteRb.isChecked() == True):
        operation='"ref_reflect"'
    return operation

def call_loadData(files,svcReflRb,svcRadRb, svcWhiteRb,tableView, spec='SVC'):
    operation=getOperation(svcReflRb,svcRadRb, svcWhiteRb, spec)
    data, model = loadTable(files,operation ,tableView)
    return data, model

def call_plotData(files,svcReflRb,svcRadRb, svcWhiteRb,spec='SVC'):
    operation=getOperation(svcReflRb,svcRadRb, svcWhiteRb, spec)
    plotGraph(files,operation)
