from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5 import QtWidgets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Ui.Univariate_RegressionUi import Ui_Form
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from sklearn import preprocessing
# from . import GdalTools_utils as Utils
POSTFIX = '_Regression'
from modules import Utils

class UnivariateRegression(Ui_Form):

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
        super(UnivariateRegression, self).setupUi(Form)
        self.Form = Form

        self.connectWidgets()

    def connectWidgets(self):
        self.lineEdit.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_5.clicked.connect(lambda: self.MetadatabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())

        self.radioButton.toggled.connect(self.toggleRadio_Linear_regression)
        self.radioButton_2.toggled.connect(self.toggleRadio_Logistic_regression)
        self.radioButton_3.toggled.connect(self.toggleRadio_Ridge_regression)
        self.radioButton_4.toggled.connect(self.toggleRadio_Lasso_regression)
        self.radioButton_5.toggled.connect(self.toggleRadio_Polynomial_regression)
        self.radioButton_6.toggled.connect(self.toggleRadio_Bayesian_regression)

        self.onlyInt = QtGui.QIntValidator()
        self.onlyDou = QtGui.QDoubleValidator()

        self.x1CoordEdit_14.setValidator(self.onlyInt)
        # self.x1CoordEdit_13.setValidator(self.onlyInt)


    def toggleRadio_Linear_regression(self):
        self.comboBox.setEnabled(self.radioButton.isChecked())

    def toggleRadio_Logistic_regression(self):
        self.comboBox.setEnabled(self.radioButton_2.isChecked())

    def toggleRadio_Ridge_regression(self):
        self.comboBox.setEnabled(self.radioButton_3.isChecked())

    def toggleRadio_Lasso_regression(self):
        self.comboBox.setEnabled(self.radioButton_4.isChecked())

    def toggleRadio_Polynomial_regression(self):
        self.comboBox.setEnabled(self.radioButton_5.isChecked())

    def toggleRadio_Bayesian_regression(self):
        self.comboBox.setEnabled(self.radioButton_6.isChecked())

    def SpectrabrowseButton_clicked(self):
        fname = Utils.browseInputFile(POSTFIX + ".csv", self.lineEdit_2, "Supported types (*.csv)", self.lineEdit_3)

        # print(self.filepath)
        if fname:
            if not Utils.validateInputFormat(fname):
                self.lineEdit_2.setText("")
                self.lineEdit_3.setText("")
                return

            self.lineEdit.setEnabled(True)
            self.pushButton_5.setEnabled(True)
            self.populateWavelength()
        else:
            self.lineEdit_2.setText("")

    def populateWavelength(self):
        self.filepath=self.lineEdit_2.text()
        df_spectra = pd.read_csv(self.filepath,
                                 header=0, index_col=0)
        wavelength = df_spectra.index.values
        self.wavelengthCmb.clear()
        for i in wavelength:
            self.wavelengthCmb.addItem(str(i))

    def populateCombobox(self):
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

            self.populateCombobox()
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

    def computeUniRegression(self, regressionType):

        try:
            self.mplWidgetSpectral_6.ax.clear()
            # band = float(self.x1CoordEdit_13.text())
            band=float(self.wavelengthCmb.currentText())
            label_one = str(self.comboBox.currentText())
            x = np.zeros((self.spectra.shape[0], 2))
            x[:, 0] = np.array(self.spectra.loc[:, band]).astype('float')
            x[:, 1] = np.array(self.data[label_one]).astype('float')
            indx = np.argsort(x[:, -1], axis=0)
            x_sort = x[indx, :]
            nr, nc = x_sort.shape
            train_per = int(self.x1CoordEdit_14.text())
            test = (100-train_per) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(x_sort[:, :-1], x_sort[:, -1], test_size=test,
                                                                random_state=42)
            # a = indx.copy()
            # b = indx[::2]
            # # train_id1 = set(a).difference(set(b))
            # train_id1 = b  # np.array(list(train_id1),dtype='int')
            #
            # diff_size = np.abs(int(nr * (train_per / 100)) - train_id1.size)
            # diff = set(a).difference(set(b))
            # diff = np.array(list(diff), dtype='int')
            #
            # c = np.random.choice(list(diff), diff_size, replace=False)
            #
            # train_id = np.hstack((train_id1, c))
            # test_id = set(a).difference(set(train_id))
            # test_id = np.array(list(test_id), dtype='int')
            #
            # X_train = x[train_id, :-1]
            # X_test = x[test_id, :-1]
            #
            # y_train = x[train_id, -1]
            # y_test = x[test_id, -1]

            if regressionType=='Linear':
                # Create linear regression object
                regr = linear_model.LinearRegression()
            elif regressionType=='Logistic':
                regr = linear_model.LogisticRegression()
                # preprocessing.LabelEncoder() - convert string or float values to 0
                lab_enc = preprocessing.LabelEncoder()
                y_train = lab_enc.fit_transform(y_train)

            elif regressionType=='Ridge':
                # Create Ridge regression object
                regr = linear_model.Ridge()
            elif regressionType=='Lasso':
                regr = linear_model.Lasso()
            elif regressionType=='Poly':
                # Create linear regression object
                poly = preprocessing.PolynomialFeatures(degree=2)
                X_train = poly.fit_transform(X_train)
                X_test = poly.fit_transform(X_test)
                poly.fit(X_train, y_train)
                # Create linear regression object
                regr = linear_model.LinearRegression()

            elif regressionType=='Bayesian':
                # Create linear regression object
                regr = linear_model.BayesianRidge()

            # Train the model using the training sets
            regr.fit(X_train, y_train)

            # Make predictions using the testing set
            y_pred = regr.predict(X_test)
            y_pred_train = regr.predict(X_train)

            # # The coefficients
            # print('Coefficients: \n', regr.coef_)
            # # The intercept
            # print('Intercept: \n', regr.intercept_)
            # The mean squared error
            trmse = mean_squared_error(y_train, y_pred_train, squared=False)
            trr2 = r2_score(y_train, y_pred_train)
            temse = mean_squared_error(y_test, y_pred, squared=False)
            ter2 = r2_score(y_test, y_pred)
            print("Test: Root Mean Squared Error :%.4f" % temse)
            print("Test: R-squared:%.4f" % ter2)
            print("Train: Root Mean Squared Error :%.4f" % trmse)
            print("Train: R-squared:%.4f" % trr2)

            result = {'y_train': y_train.flatten(), 'y_test': y_test.flatten(),
                      'y_pred': y_pred.flatten(), 'R2': [r2_score(y_test, y_pred)],
                      'MSE:': [mean_squared_error(y_test, y_pred,squared=False)]}
            df = [pd.DataFrame({k: v}) for k, v in result.items()]
            df = pd.concat(df, axis=1)
            # print(df)
            df.to_csv(self.outputFilename, index=False)

            # Plot outputs
            with plt.style.context('ggplot'):
                self.mplWidgetSpectral_6.ax.scatter(y_train, y_pred_train, color='black', label="Calibration")
                z = np.polyfit(y_train, y_pred_train, 1)
                p = np.poly1d(z)
                self.mplWidgetSpectral_6.ax.plot(y_train, y_train, color='green', label="Expected")
                self.mplWidgetSpectral_6.ax.scatter(y_test, y_pred, color='red', label="Validation")
                z = np.polyfit(y_test, y_pred, 1)
                p = np.poly1d(z)

                # self.mplWidgetSpectral_9.ax.plot(np.polyval(z, y_test), y_test, color='blue', label='Predicted regression line')
                self.mplWidgetSpectral_6.ax.plot(y_test, p(y_test), color='blue', label='Predicted')
                self.mplWidgetSpectral_6.ax.set_xlabel('Observed')
                self.mplWidgetSpectral_6.ax.set_ylabel('Predicted')
                # self.mplWidgetSpectral_6.ax.legend()
                legend = self.mplWidgetSpectral_6.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                legend.set_draggable(True)

                self.mplWidgetSpectral_6.canvas.draw()
            # self.mplWidgetSpectral_6.ax.scatter(X_test, y_test, color='black',label='X_test vs y_test')
            # self.mplWidgetSpectral_6.ax.scatter(X_test, y_pred, color='blue',label='X_test vs y_pred')
            # self.mplWidgetSpectral_6.ax.set_xlabel('Observed')
            # self.mplWidgetSpectral_6.ax.set_ylabel("Predicted")
            # self.mplWidgetSpectral_6.ax.legend()
            # self.mplWidgetSpectral_6.canvas.draw()
        except Exception as e:
            import traceback
            print(e, traceback.format_exc())





    def run(self):
        #################### Validation ##########################
        if (self.lineEdit_2.text() is None) or (self.lineEdit_2.text() == ""):
            self.lineEdit_2.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit_2.text())):
            self.lineEdit_2.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                           "Kindly enter a valid input file.",
                                           QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit.text() is None) or (self.lineEdit.text() == ""):
            self.lineEdit.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Input File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.exists(self.lineEdit.text())):
            self.lineEdit.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid", "Kindly enter a valid file.",
                                           QtWidgets.QMessageBox.Ok)
            return

        if (self.lineEdit_3.text() is None) or (self.lineEdit_3.text() == ""):
            self.lineEdit_3.setFocus()
            QtWidgets.QMessageBox.warning(self.Form, 'Information missing or invalid', "Output File is required",
                                          QtWidgets.QMessageBox.Ok)
            return

        if (not os.path.isdir(os.path.dirname(self.lineEdit_3.text()))):
            self.lineEdit_3.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                           "Kindly enter a valid output path.",
                                           QtWidgets.QMessageBox.Ok)
            return

        if self.comboBox.currentText()=="--Select--":
            self.comboBox.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                           "Select the property first",
                                           QtWidgets.QMessageBox.Ok)
            return

        if int(self.x1CoordEdit_14.text())>=100 or int(self.x1CoordEdit_14.text())<=0 :
            self.x1CoordEdit_14.setFocus()
            QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                           "Invalid input: sample size cannot be >=100 or <=0  ",
                                           QtWidgets.QMessageBox.Ok)
            return



        try:



            self.filepath = self.lineEdit_2.text()
            self.outputFilename = self.lineEdit_3.text()
            self.metafilepath = self.lineEdit.text()

            print("In: " + self.filepath)
            print("Out: " + self.outputFilename)
            print("Meta data: " + self.metafilepath)
            print("Running...")


            df_spectra = pd.read_csv(self.filepath,
                                     header=0, index_col=0)
            wavelength = df_spectra.index.values
            if (int(self.wavelengthCmb.currentText()) < wavelength[0]) or (int(self.wavelengthCmb.currentText()) > wavelength[-1]):
                self.wavelengthCmb.setFocus()
                QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                               "Invalid input wavelength",
                                               QtWidgets.QMessageBox.Ok)
                return

            df_metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
            self.data = df_metadata.T


            # -----------------------------------------------------------------------------------------------------
            self.spectra = df_spectra.T
            band = float(self.wavelengthCmb.currentText())
            label_one = str(self.comboBox.currentText())
            x = np.zeros((self.spectra.shape[0], 2))
            x[:, 0] = np.array(self.spectra.loc[:, band]).astype('float')
            x[:, 1] = np.array(self.data[label_one]).astype('float')
            if np.isnan(x).any():
                QtWidgets.QMessageBox.warning(self.Form, 'Error',
                                              "Data may contain invalid values like nan for example.",
                                              QtWidgets.QMessageBox.Ok)
                return

            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            if self.radioButton.isChecked():
                self.computeUniRegression('Linear')

            if self.radioButton_2.isChecked():
                self.computeUniRegression('Logistic')
            if self.radioButton_3.isChecked():
                self.computeUniRegression('Ridge')
            if self.radioButton_4.isChecked():
                self.computeUniRegression('Lasso')
            if self.radioButton_5.isChecked():
                self.computeUniRegression('Poly')
            if self.radioButton_6.isChecked():
                self.computeUniRegression('Bayesian')

            QApplication.restoreOverrideCursor()

        except Exception as e:
            import traceback
            print(e, traceback.format_exc())
            QApplication.restoreOverrideCursor()
