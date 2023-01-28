
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5 import QtWidgets
import math
import sklearn.model_selection as ms
from Ui.Multivariate_RegressionUi import Ui_Form
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QApplication,QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from sklearn import preprocessing
import scipy.signal._savitzky_golay as savgol
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# compare ensemble to each standalone models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

import itertools

# from . import GdalTools_utils as Utils
POSTFIX = '_Regression'

from modules import Utils

class Multi_Regression(Ui_Form):

    def __init__(self):
        self.curdir = None
        self.filepath = []
        self.outputFilename = ""
        self.data = None
        self.metadata=None


    def get_widget(self):
        return self.groupBox

    def isEnabled(self):
        """
        Checks to see if current widget isEnabled or not
        :return:
        """
        return self.get_widget().isEnabled()

    def k_fold_change_testfolds(self, testSb,crossSb):
        if self.data is not None:
            spectra=self.data.iloc[:, 1:].to_numpy()
            # print(spectra.shape)
            n_samples =int((100 - testSb.value()) / 100 * spectra.shape[1])
            # print(n_samples, testSb.value())
            crossSb.setMaximum(n_samples)

    def setupUi(self, Form):
        super(Multi_Regression, self).setupUi(Form)
        # self.tabWidget_4.removeTab(1)
        self.Form = Form

        self.crossValSb.setMinimum(1)
        self.crossValSb.setMaximum(99)
        self.crossValSb.setValue(10)

        self.testSizeSb.setMinimum(1)
        self.testSizeSb.setMaximum(99)
        self.testSizeSb.setValue(70)

        self.crossValStackedSb.setMinimum(1)
        self.crossValStackedSb.setMaximum(99)
        self.crossValStackedSb.setValue(10)

        self.testSizeStackedSb.setMinimum(1)
        self.testSizeStackedSb.setMaximum(99)
        self.testSizeStackedSb.setValue(70)

        self.crossValSvSb.setMinimum(1)
        self.crossValSvSb.setMaximum(99)
        self.crossValSvSb.setValue(10)

        self.testSizeSvSb.setMinimum(1)
        self.testSizeSvSb.setMaximum(99)
        self.testSizeSvSb.setValue(70)

        self.testSizeSb.valueChanged.connect(lambda:self.k_fold_change_testfolds(self.testSizeSb, self.crossValSb))
        self.testSizeStackedSb.valueChanged.connect(lambda:self.k_fold_change_testfolds(self.testSizeStackedSb, self.crossValStackedSb))
        self.testSizeSvSb.valueChanged.connect(lambda:self.k_fold_change_testfolds(self.testSizeSvSb, self.crossValSvSb))


        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton_4.clicked.connect(lambda: self.SpectrabrowseButton_clicked())
        self.pushButton_5.clicked.connect(lambda: self.MetadatabrowseButton_clicked())
        self.pushButton_6.clicked.connect(lambda: self.saveas())

        self.lineEdit.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.populateKernal()

        self.onlyInt = QtGui.QIntValidator()
        self.onlyDou = QtGui.QDoubleValidator()
        self.tabWidget_4.setCurrentIndex(0)


    def populateCombobox(self, combobox):
        self.metafilepath=self.lineEdit.text()
        df_metadata = pd.read_csv(self.metafilepath, header=None, index_col=0)
        self.metadata = df_metadata.T

        combobox.clear()
        combobox.addItem("--Select--")
        value = self.metadata.drop(["Spectra ID"], axis=1)
        for i in value.columns:
            combobox.addItem(str(i))

    def populateKernal(self):
        kernels = ['rbf', 'poly', 'linear']
        self.comboBox_2.clear()
        for item in kernels:
            self.comboBox_2.addItem(str(item))


    def SpectrabrowseButton_clicked(self):
        fname=Utils.browseInputFile(POSTFIX + ".csv", self.lineEdit_2, "Supported types (*.csv)", self.lineEdit_3)

        # print(self.filepath)
        if fname:
            if not Utils.validateInputFormat(fname):
                self.lineEdit_2.setText("")
                self.lineEdit_3.setText("")
                return


            self.lineEdit.setEnabled(True)
            self.pushButton_5.setEnabled(True)

            self.data = pd.read_csv(fname, header=0, index_col=None)
            spectra = self.data.iloc[:, 1:].to_numpy()
            # print(spectra.shape)

            n_samples=(100-self.testSizeSb.value())/100 * spectra.shape[1]
            max_component=min(spectra.shape[0],spectra.shape[1])

            self.componentSb.setMinimum(1)
            self.componentSb.setMaximum(max_component)
            self.componentSb.setValue(20)

            self.compStackedSb.setMinimum(1)
            self.compStackedSb.setMaximum(max_component)
            self.compStackedSb.setValue(20)

            self.crossValSb.setMaximum(n_samples)
            self.crossValStackedSb.setMaximum(n_samples)
            self.crossValSvSb.setMaximum(n_samples)

        else:
            self.lineEdit_2.setText("")

    def MetadatabrowseButton_clicked(self):

        fname = Utils.browseMetadataFile(self.lineEdit, "Supported types (*.csv)")


        if fname:
            if not Utils.validateMetaFormat(self.lineEdit_2.text(),fname):
                self.lineEdit.setText("")
                return

            self.populateCombobox(self.comboBox)
            self.populateCombobox(self.comboBox_5)
            self.populateCombobox(self.comboBox_6)
        else:
            self.lineEdit.setText("")
            self.comboBox.clear()
            self.comboBox_5.clear()
            self.comboBox_6.clear()

    def saveas(self):
        self.outputFilename, _ = QFileDialog.getSaveFileName(None, 'save', self.curdir, '*.csv')
        if self.outputFilename:
            self.lineEdit_3.setText(self.outputFilename)

        return self.outputFilename
        # Plot the mses

    def computeRegression(self, transformOption, metaCombobox, trainSB, normalize,regression):
        self.mplWidgetSpectral_6.ax.clear()
        self.mplWidgetSpectral_9.ax.clear()
        self.mplWidgetSpectral_10.ax.clear()
        try:
            if transformOption == "log":
                print('Preprocess : Transformed Log')
                X = 1 / np.log(self.df_spectra)

            elif transformOption == "std":
                print('Preprocess : Transform Standard Scalar')
                scale = preprocessing.StandardScaler()
                # scale_clay.mean_  , scale_clay.transform()
                scale_fit = scale.fit(self.df_spectra.T)
                X = scale_fit.transform(self.df_spectra.T).T

            elif transformOption == 'norm':
                print('Preprocess : Transform Standard Normal Variate')
                scale = preprocessing.StandardScaler()
                # scale_clay.mean_  , scale_clay.transform()
                scale_fit = scale.fit(self.df_spectra)
                X = scale_fit.transform(self.df_spectra)


            elif transformOption == 'sav':
                print('Preprocess : Savitsky-Golay')
                X = savgol.savgol_filter(self.df_spectra, 9, 2, deriv=0, axis=0, mode='nearest')

            elif transformOption == "smooth":
                print('Preprocess : Non Smoothing')
                X = self.df_spectra

            label_one = str(metaCombobox.currentText())
            Meta_Y = np.array(self.df_metadata[label_one]).astype('float')
            x = np.zeros((X.shape[0], X.shape[1] + 1))
            x[:, :-1] = X
            x[:, -1] = np.array(self.df_metadata[label_one]).astype('float')
            if normalize.isChecked():
                x[:, -1] = stats.boxcox(Meta_Y)[0]
            else:
                x[:, -1] = Meta_Y

            indx = np.argsort(x[:, -1], axis=0)
            x_sort = x[indx, :]
            nr, nc = x_sort.shape
            test = int(trainSB.value()) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(x_sort[:, :-1], x_sort[:, -1], test_size=test,
                                                                random_state=42)

            # return X_train, y_train, X_test, y_test
            flag=False
            if regression=='PLS' and  self.radioButton_8.isChecked() :
                # test with components
                r2s = []
                mses = []
                rpds = []
                component_number = int(self.componentSb.value())
                xticks = np.arange(1, component_number)
                for n_comp in xticks:
                    y_cv, r2, mse, rpd = self.optimise_pls_cv(X_train, y_train, n_comp)
                    r2s.append(r2)
                    mses.append(mse)
                    rpds.append(rpd)
                self.plot_metrics(mses, 'MSE', 'min')
                # self.plot_metrics(rpds, 'RPD', 'max')
                # self.plot_metrics(r2s, 'R2', 'max')
                print('R2: %f, MSE: %f, RPD: %f' % (r2, mse, rpd))
                flag = True

            elif regression=='PLS' and self.radioButton_9.isChecked():
                    component_number = int(self.componentSb.value())
                    pls = PLSRegression(n_components=component_number)
                    pls.fit(X_train, y_train)
                    y_pred = pls.predict(X_test)
                    y_pred_train = pls.predict(X_train)
                    self.predict_plot_metrics(y_test, y_pred, y_train, y_pred_train)
                    # print('R2:', r2_score(y_test, y_pred))
                    # print('MSE:', mean_squared_error(y_test, y_pred))
                    #
                    # result = {'y_train': y_train.flatten(), 'y_pred_train': y_pred_train.flatten(),
                    #           'y_test': y_test.flatten(), 'y_pred': y_pred.flatten(), 'R2': [r2_score(y_test, y_pred)],
                    #           'MSE:': [mean_squared_error(y_test, y_pred)]}
                    # df = [pd.DataFrame({k: v}) for k, v in result.items()]
                    # df = pd.concat(df, axis=1)
                    # # print(df)
                    # df.to_csv(self.outputFilename, index=False)

                    print('Results saved successfully !!')
            elif regression=='Stacked':
                component_number = int(self.compStackedSb.value())
                cross_validation_number = int(self.crossValStackedSb.value())
                clf = self.get_stacking(component_number)
                scores = self.evaluate_model(clf, X_train, y_train, cross_validation_number)
                # print('>%s %.3f (%.3f)' % ('Stacked Regression : ', mean(scores), std(scores)))
                clf.fit(X_train, y_train)
                y_pred_train = clf.predict(X_train)
                y_pred = clf.predict(X_test)
                self.plot_stacked_regression(y_test, y_pred, y_train, y_pred_train)
                # mse = mean_squared_error(y_test, y_pred)
                # r2 = r2_score(y_test, y_pred)
                # print(y_test.mean(), y_pred.mean())
                # print("Mean Squared Error", mse)
                # print("R-squared:", r2)
                #
                # result = {'y_train': y_train.flatten(), 'y_pred_train': y_pred_train.flatten(),
                #           'y_test': y_test.flatten(),
                #           'y_pred': y_pred.flatten(), 'R2': [r2_score(y_test, y_pred)],
                #           'MSE:': [mean_squared_error(y_test, y_pred)]}
                # df = [pd.DataFrame({k: v}) for k, v in result.items()]
                # df = pd.concat(df, axis=1)
                # # print(df)
                # df.to_csv(self.outputFilename, index=False)

                print('Results saved successfully !!')

            elif regression=='SVR':
                clf = self.support_optimise_pls_cv(X_train, y_train)
                y_pred_train = clf.predict(X_train)
                y_pred = clf.predict(X_test)
                self.support_predict_plot_metrics(y_test, y_pred, y_train, y_pred_train)
                # mse = mean_squared_error(y_test, y_pred)
                # r2 = r2_score(y_test, y_pred)
                # self.support_predict_plot_metrics(y_test, y_pred, y_train, y_pred_train)
                # print(y_test.mean(), y_pred.mean())
                # print("Mean Squared Error", mse)
                # print("R-squared:", r2)
                #
                # result = {'y_train': y_train.flatten(), 'y_pred_train': y_pred_train.flatten(),
                #           'y_test': y_test.flatten(),
                #           'y_pred': y_pred.flatten(), 'R2': [r2_score(y_test, y_pred)],
                #           'MSE:': [mean_squared_error(y_test, y_pred)]}
                # df = [pd.DataFrame({k: v}) for k, v in result.items()]
                # df = pd.concat(df, axis=1)
                # # print(df)
                # df.to_csv(self.outputFilename, index=False)
                #
                # print('Results saved successfully !!')
            if not flag:
                trmse = mean_squared_error(y_train, y_pred_train, squared=False)
                trr2 = r2_score(y_train, y_pred_train)
                temse = mean_squared_error(y_test, y_pred, squared=False)
                ter2 = r2_score(y_test, y_pred)

                # print(y_test.mean(), y_pred.mean())
                print("Test: Root Mean Squared Error :%.4f" % temse)
                print("Test: R-squared:%.4f" % ter2)
                print("Train: Root Mean Squared Error :%.4f" % trmse)
                print("Train: R-squared:%.4f" % trr2)

                result = {'y_train': y_train.flatten(), 'y_pred_train': y_pred_train.flatten(),
                          'y_test': y_test.flatten(),
                          'y_pred': y_pred.flatten(), 'R2': [r2_score(y_test, y_pred)],
                          'MSE:': [mean_squared_error(y_test, y_pred, squared=False)]}
                df = [pd.DataFrame({k: v}) for k, v in result.items()]
                df = pd.concat(df, axis=1)
                # print(df)
                df.to_csv(self.outputFilename, index=False)

                print('Results saved successfully !!')


        except Exception as e:
            import traceback
            print(e, traceback.format_exc())



    def optimise_pls_cv(self, X, y, n_comp):
        # Define PLS object
        pls = PLSRegression(n_components=n_comp)

        # Cross-validation
        cross_validation_number = int(self.crossValSb.value())
        y_cv = cross_val_predict(pls, X, y, cv=cross_validation_number)

        # Calculate scores
        r2 = r2_score(y, y_cv)
        mse = mean_squared_error(y, y_cv)
        rpd = y.std() / np.sqrt(mse)
        return (y_cv, r2, mse, rpd)

    def plot_metrics(self, vals, ylabel, objective):
        self.mplWidgetSpectral_6.ax.clear()
        component_number = int(self.componentSb.value())
        xticks = np.arange(1, component_number)
        with plt.style.context('ggplot'):
            self.mplWidgetSpectral_6.ax.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
            if objective == 'min':
                idx = np.argmin(vals)
            else:
                idx = np.argmax(vals)
            self.mplWidgetSpectral_6.ax.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

            self.mplWidgetSpectral_6.ax.set_xlabel('Number of PLS components')
            self.mplWidgetSpectral_6.ax.set_xticks = xticks
            self.mplWidgetSpectral_6.ax.set_ylabel(ylabel)
            self.mplWidgetSpectral_6.ax.set_title('PLS')
            self.mplWidgetSpectral_6.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

        self.mplWidgetSpectral_6.canvas.draw()

    def predict_plot_metrics(self, y_test, y_pred, y_train, y_pred_train):
        self.mplWidgetSpectral_6.ax.clear()
        self.mplWidgetSpectral_10.ax.clear()
        self.mplWidgetSpectral_9.ax.clear()
        with plt.style.context('ggplot'):
            self.mplWidgetSpectral_6.ax.scatter(y_train, y_pred_train, color='black', label="Calibration")
            # z = np.polyfit(y_train, y_pred_train, 1)
            # p = np.poly1d(z)
            self.mplWidgetSpectral_6.ax.plot(y_train, y_train, color='green', label="Expected")
            self.mplWidgetSpectral_6.ax.scatter(y_test, y_pred, color='red', label="Validation")
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(np.squeeze(z))

            # self.mplWidgetSpectral_9.ax.plot(np.polyval(z, y_test), y_test, color='blue', label='Predicted regression line')
            self.mplWidgetSpectral_6.ax.plot(y_test, p(y_test), color='blue', label='Predicted')
            self.mplWidgetSpectral_6.ax.set_xlabel('Observed')
            self.mplWidgetSpectral_6.ax.set_ylabel('Predicted')
            self.mplWidgetSpectral_6.ax.legend()
            self.mplWidgetSpectral_6.canvas.draw()

            # get a stacking ensemble of models

    def get_stacking(self,n_component):
        # define the base models
        level0 = list()
        level0.append(('knn', KNeighborsRegressor()))
        level0.append(('cart', DecisionTreeRegressor()))
        level0.append(('svm', SVR()))
        level0.append(('pls', PLSRegression(n_components=n_component)))
        # define meta learner model
        level1 = LinearRegression()
        # define the stacking ensemble
        model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
        return model

    # evaluate a given model using cross-validation
    def evaluate_model(self,model, X, y, n_split):
        cv = RepeatedKFold(n_splits=n_split, n_repeats=10, random_state=1)
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,
                                 error_score='raise')
        return scores

    def plot_stacked_regression(self,y_test, y_pred, y_train, y_pred_train):
        self.mplWidgetSpectral_6.ax.clear()
        self.mplWidgetSpectral_10.ax.clear()
        self.mplWidgetSpectral_9.ax.clear()
        # # min_x = np.hstack((y_test, y_pred)).min()
        # # x_max = np.hstack((y_test, y_pred)).max()
        # # x = np.linspace(min_x, x_max, np.size(y_pred))
        # self.mplWidgetSpectral_10.ax.plot(y_test, y_pred, 'ro', x, x, "-")
        # self.mplWidgetSpectral_10.ax.set_xlabel('Observed')
        # self.mplWidgetSpectral_10.ax.set_ylabel('Predicted')
        # self.mplWidgetSpectral_10.canvas.draw()
        with plt.style.context('ggplot'):
            self.mplWidgetSpectral_9.ax.scatter(y_train, y_pred_train, color='black', label="Calibration")
            z = np.polyfit(y_train, y_pred_train, 1)
            p = np.poly1d(z)
            self.mplWidgetSpectral_9.ax.plot(y_train, y_train, color='green', label="Expected")
            self.mplWidgetSpectral_9.ax.scatter(y_test, y_pred, color='red', label="Validation")
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)

            # self.mplWidgetSpectral_9.ax.plot(np.polyval(z, y_test), y_test, color='blue', label='Predicted regression line')
            self.mplWidgetSpectral_9.ax.plot(y_test, p(y_test), color='blue', label='Predicted')
            self.mplWidgetSpectral_9.ax.set_xlabel('Observed')
            self.mplWidgetSpectral_9.ax.set_ylabel('Predicted')
            self.mplWidgetSpectral_9.ax.legend()
            self.mplWidgetSpectral_9.canvas.draw()



    def SecondCombo(self, i):
        labels = self.kernel
        self.comboBox_2.clear()
        for item in labels:
            self.comboBox_2.addItem(str(item))

    def support_optimise_pls_cv(self, X, y):
        self.mplWidgetSpectral_10.ax.clear()

        # Select the kernal
        kernal = str(self.comboBox_2.currentText())
        if kernal == 'rbf':
            parameters = {'kernel': ['rbf'], "gamma": [1e-4, 1e-3, 0.001, 0.1, 0.2, 0.5, 0.6, 0.9],
                          'C': [1, 10, 100, 1000, 10000]}
        elif kernal == 'poly':
            parameters = {'kernel': ['poly'], "gamma": [1e-4, 1e-3, 0.001, 0.1, 0.2, 0.5, 0.6, 0.9],
                          'C': [1, 10, 100, 1000, 10000], 'degree': [3]}
        else:
            parameters = {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 10000]}
        scorer = make_scorer(mean_squared_error)
        supportvector = SVR(kernel=kernal)

        # Cross-validation
        cv = int(self.crossValSvSb.value())
        clf = GridSearchCV(supportvector, parameters, scoring=scorer, cv=cv, return_train_score=False)
        clf.fit(X,y)
        print(clf.best_params_)
        return clf


    def support_predict_plot_metrics(self, y_test, y_pred,y_train,y_pred_train):
        self.mplWidgetSpectral_6.ax.clear()
        self.mplWidgetSpectral_10.ax.clear()
        self.mplWidgetSpectral_9.ax.clear()

        with plt.style.context('ggplot'):
            self.mplWidgetSpectral_10.ax.scatter(y_train, y_pred_train, color='black', label="Calibration")
            z = np.polyfit(y_train, y_pred_train, 1)
            p = np.poly1d(z)
            self.mplWidgetSpectral_10.ax.plot(y_train, y_train, color='green', label="Expected")
            self.mplWidgetSpectral_10.ax.scatter(y_test, y_pred, color='red', label="Validation")
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)

            # self.mplWidgetSpectral_9.ax.plot(np.polyval(z, y_test), y_test, color='blue', label='Predicted regression line')
            self.mplWidgetSpectral_10.ax.plot(y_test, p(y_test), color='blue', label='Predicted')
            self.mplWidgetSpectral_10.ax.set_xlabel('Observed')
            self.mplWidgetSpectral_10.ax.set_ylabel('Predicted')
            self.mplWidgetSpectral_10.ax.legend()
            self.mplWidgetSpectral_10.canvas.draw()

    def run(self):

        if not (Utils.validataInputPath(self.lineEdit_2, self.Form) and \
                Utils.validataOutputPath(self.lineEdit_3, self.Form) and \
                Utils.validateExtension(self.lineEdit_2, 'csv', self.Form) and \
                Utils.validateExtension(self.lineEdit_3, 'csv', self.Form) and \
                Utils.validateEmpty(self.lineEdit, self.Form)
                ):
            return

        try:

            self.inFile = self.lineEdit_2.text()
            self.outputFilename = self.lineEdit_3.text()
            self.Metafile = self.lineEdit.text()

            print("In: " + self.inFile)
            print("Out: " + self.outputFilename)
            print("Meta data: " + self.Metafile)
            print("Running...")

            data = pd.read_csv(self.inFile, header=0, index_col=None)
            metadata = pd.read_csv(self.Metafile, header=None, index_col=0)
            spectra = data.iloc[:, 1:].to_numpy()
            self.df_spectra = spectra.T
            self.df_metadata = metadata.T

            if self.tabWidget_4.currentIndex()==0:

                if self.comboBox.currentText() == "--Select--":
                    self.comboBox.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Select the property first",
                                                   QtWidgets.QMessageBox.Ok)
                    return



                if not self.radioButton_5.isChecked() and not self.radioButton_6.isChecked():
                    self.radioButton_5.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Please select normalization option first",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if not self.radioButton_8.isChecked() and not self.radioButton_9.isChecked():
                    self.radioButton_8.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Please select validation and testing option first",
                                                   QtWidgets.QMessageBox.Ok)
                    return



                if int(self.componentSb.value()) > spectra.shape[0] or int(self.componentSb.value()) <= 0:
                    self.componentSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Number of components cannot be <=0 or >" + str(spectra.shape[0]),
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if int(self.crossValSb.value()) >= spectra.shape[1] or int(self.crossValSb.value()) <= 0:
                    self.crossValSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Number of split cannot be <=0 or >" + str(spectra.shape[1]),
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if int(self.testSizeSb.value()) >= 100 or int(self.testSizeSb.value()) <= 0:
                    self.testSizeSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Test size cannot be >=100 or <=0",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
                # ------------- Spectral Transform 1/Log------------------------
                if self.radioButton.isChecked():


                    self.computeRegression('log', self.comboBox,
                                                self.testSizeSb,
                                                self.radioButton_5,'PLS')



                # ------------- Spectral Transform Standard Scalar------------------------
                if self.radioButton_2.isChecked():
                    self.computeRegression('std', self.comboBox,
                                           self.testSizeSb,
                                           self.radioButton_5, 'PLS')


                # -------------- Spectral Transform Standard Normal Variate --------------------#
                if self.radioButton_3.isChecked():
                    self.computeRegression('norm', self.comboBox,
                                           self.testSizeSb,
                                           self.radioButton_5, 'PLS')


                # -----------Spectral Savitsky_Golay -----------------#
                if self.radioButton_4.isChecked():
                    self.computeRegression('sav', self.comboBox,
                                           self.testSizeSb,
                                           self.radioButton_5, 'PLS')


                # -----------Non Smooting spectra -----------------#
                if self.radioButton_7.isChecked():
                    self.computeRegression('smooth', self.comboBox,
                                           self.testSizeSb,
                                           self.radioButton_5, 'PLS')

                QApplication.restoreOverrideCursor()
            if self.tabWidget_4.currentIndex() == 1:


                if self.comboBox_5.currentText() == "--Select--":
                    self.comboBox_5.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Select the property first",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if not self.radioButton_42.isChecked() and not self.radioButton_43.isChecked():
                    self.radioButton_42.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Please select normalization option first",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if int(self.compStackedSb.value()) > spectra.shape[0] or int(self.compStackedSb.value()) <= 0:
                    self.compStackedSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Number of components cannot be <=0 or >" + str(spectra.shape[0]),
                                                   QtWidgets.QMessageBox.Ok)
                    return


                if int(self.crossValStackedSb.value()) >= spectra.shape[1] or int(self.crossValStackedSb.value()) <= 0:
                    self.crossValStackedSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Number of split cannot be <=0 or >" + str(spectra.shape[1]),
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if int(self.testSizeStackedSb.value()) >= 100 or int(self.testSizeStackedSb.value()) <= 0:
                    self.testSizeStackedSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Test size cannot be >=100 or <=0",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

                # ------------- Spectral Transform 1/Log------------------------
                if self.radioButton_37.isChecked():
                    self.computeRegression('log', self.comboBox_5,
                                           self.testSizeStackedSb,
                                           self.radioButton_42, 'Stacked')


                if self.radioButton_38.isChecked():
                    self.computeRegression('std', self.comboBox_5,
                                           self.testSizeStackedSb,
                                           self.radioButton_42, 'Stacked')


                # -------------- Spectral Transform Standard Normal Variate --------------------#
                if self.radioButton_39.isChecked():
                    self.computeRegression('norm', self.comboBox_5,
                                           self.testSizeStackedSb,
                                           self.radioButton_42, 'Stacked')

                # -----------Spectral Savitsky_Golay -----------------#
                if self.radioButton_40.isChecked():
                    self.computeRegression('sav', self.comboBox_5,
                                           self.testSizeStackedSb,
                                           self.radioButton_42, 'Stacked')

                # -----------Non Smooting spectra -----------------#
                if self.radioButton_41.isChecked():
                    self.computeRegression('smooth', self.comboBox_5,
                                           self.testSizeStackedSb,
                                           self.radioButton_42, 'Stacked')

                QApplication.restoreOverrideCursor()

            if self.tabWidget_4.currentIndex() == 2:

                if self.comboBox_6.currentText() == "--Select--":
                    self.comboBox_6.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Select the property first",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if not self.radioButton_51.isChecked() and not self.radioButton_52.isChecked():
                    self.radioButton_51.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Please select normalization option first",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                # if not self.radioButton_53.isChecked() and not self.radioButton_54.isChecked():
                #     self.radioButton_53.setFocus()
                #     QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                #                                    "Please select validation and testing option first",
                #                                    QtWidgets.QMessageBox.Ok)
                #     return

                if int(self.crossValSvSb.value()) >= spectra.shape[1] or int(self.crossValSvSb.value()) <= 0:
                    self.crossValSvSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Number of split cannot be <=0 or >" + str(spectra.shape[1]),
                                                   QtWidgets.QMessageBox.Ok)
                    return

                if int(self.testSizeSvSb.value()) >= 100 or int(self.testSizeSvSb.value()) <= 0:
                    self.testSizeSvSb.setFocus()
                    QtWidgets.QMessageBox.critical(self.Form, "Information missing or invalid",
                                                   "Test size cannot be >=100 or <=0",
                                                   QtWidgets.QMessageBox.Ok)
                    return

                QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

                # ------------- Spectral Transform 1/Log------------------------
                if self.radioButton_46.isChecked():
                    self.computeRegression('log', self.comboBox_6,
                                           self.testSizeSvSb,
                                           self.radioButton_51, 'SVR')
                # ------------- Spectral Transform Standard Scalar------------------------
                if self.radioButton_47.isChecked():
                    self.computeRegression('std', self.comboBox_6,
                                           self.testSizeSvSb,
                                           self.radioButton_51, 'SVR')
                # -------------- Spectral Transform Standard Normal Variate --------------------#
                if self.radioButton_48.isChecked():
                    self.computeRegression('norm', self.comboBox_6,
                                           self.testSizeSvSb,
                                           self.radioButton_51, 'SVR')
                # -----------Spectral Savitsky_Golay -----------------#
                if self.radioButton_49.isChecked():
                    self.computeRegression('sav', self.comboBox_6,
                                           self.testSizeSvSb,
                                           self.radioButton_51, 'SVR')
                # -----------Non Smooting spectra -----------------#
                if self.radioButton_50.isChecked():
                    self.computeRegression('smooth', self.comboBox_6,
                                           self.testSizeSvSb,
                                           self.radioButton_51, 'SVR')

                QApplication.restoreOverrideCursor()

        except Exception as e:
            import traceback
            print(e, traceback.format_exc())
            QApplication.restoreOverrideCursor()
