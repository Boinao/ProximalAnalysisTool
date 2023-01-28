# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainwindowUi.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1120, 771)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.splitter.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setHandleWidth(10)
        self.splitter.setObjectName("splitter")
        self.scrollArea = QtWidgets.QScrollArea(self.splitter)
        self.scrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scrollArea.setFrameShadow(QtWidgets.QFrame.Plain)
        self.scrollArea.setLineWidth(0)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(QtCore.Qt.AlignCenter)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1085, 193))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout.setObjectName("gridLayout")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.textBrowser = QtWidgets.QTextBrowser(self.splitter)
        self.textBrowser.setEnabled(True)
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 150))
        self.textBrowser.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_3.addWidget(self.splitter)
        self.progress_OK = QtWidgets.QHBoxLayout()
        self.progress_OK.setObjectName("progress_OK")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progress_OK.addWidget(self.progressBar)
        self.okPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.okPushButton.setObjectName("okPushButton")
        self.progress_OK.addWidget(self.okPushButton)
        self.stopPushButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopPushButton.setObjectName("stopPushButton")
        self.progress_OK.addWidget(self.stopPushButton)
        self.deletePushButton = QtWidgets.QPushButton(self.centralwidget)
        self.deletePushButton.setObjectName("deletePushButton")
        self.progress_OK.addWidget(self.deletePushButton)
        self.verticalLayout_3.addLayout(self.progress_OK)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1120, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuPreprocessing = QtWidgets.QMenu(self.menubar)
        self.menuPreprocessing.setObjectName("menuPreprocessing")
        self.menuTransform = QtWidgets.QMenu(self.menubar)
        self.menuTransform.setObjectName("menuTransform")
        self.menuRegression = QtWidgets.QMenu(self.menubar)
        self.menuRegression.setObjectName("menuRegression")
        self.menuFeature_Selectionj = QtWidgets.QMenu(self.menubar)
        self.menuFeature_Selectionj.setObjectName("menuFeature_Selectionj")
        self.menuDimensionality_Reduction = QtWidgets.QMenu(self.menubar)
        self.menuDimensionality_Reduction.setObjectName("menuDimensionality_Reduction")
        self.menuClassification = QtWidgets.QMenu(self.menubar)
        self.menuClassification.setObjectName("menuClassification")
        self.menuVisualizer = QtWidgets.QMenu(self.menubar)
        self.menuVisualizer.setObjectName("menuVisualizer")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Preprocessing = QtWidgets.QAction(MainWindow)
        self.actionLoad_Preprocessing.setObjectName("actionLoad_Preprocessing")
        self.actionLoad_Data = QtWidgets.QAction(MainWindow)
        self.actionLoad_Data.setObjectName("actionLoad_Data")
        self.actionLoad_Transform = QtWidgets.QAction(MainWindow)
        self.actionLoad_Transform.setObjectName("actionLoad_Transform")
        self.actionLoad_Regression = QtWidgets.QAction(MainWindow)
        self.actionLoad_Regression.setObjectName("actionLoad_Regression")
        self.actionStratified = QtWidgets.QAction(MainWindow)
        self.actionStratified.setObjectName("actionStratified")
        self.actionCross_Validation = QtWidgets.QAction(MainWindow)
        self.actionCross_Validation.setObjectName("actionCross_Validation")
        self.actionSpectral_Indices = QtWidgets.QAction(MainWindow)
        self.actionSpectral_Indices.setObjectName("actionSpectral_Indices")
        self.actionFeature_Selection = QtWidgets.QAction(MainWindow)
        self.actionFeature_Selection.setObjectName("actionFeature_Selection")
        self.actionMultiple_Classes = QtWidgets.QAction(MainWindow)
        self.actionMultiple_Classes.setObjectName("actionMultiple_Classes")
        self.actionDimension_Reduction = QtWidgets.QAction(MainWindow)
        self.actionDimension_Reduction.setObjectName("actionDimension_Reduction")
        self.actionSpectral_Distance = QtWidgets.QAction(MainWindow)
        self.actionSpectral_Distance.setObjectName("actionSpectral_Distance")
        self.actionClassification = QtWidgets.QAction(MainWindow)
        self.actionClassification.setObjectName("actionClassification")
        self.actionVisualizer = QtWidgets.QAction(MainWindow)
        self.actionVisualizer.setObjectName("actionVisualizer")
        self.actionView_Data = QtWidgets.QAction(MainWindow)
        self.actionView_Data.setObjectName("actionView_Data")
        self.actionGrid_Search = QtWidgets.QAction(MainWindow)
        self.actionGrid_Search.setObjectName("actionGrid_Search")
        self.actionResampling = QtWidgets.QAction(MainWindow)
        self.actionResampling.setObjectName("actionResampling")
        self.actionDerivative = QtWidgets.QAction(MainWindow)
        self.actionDerivative.setObjectName("actionDerivative")
        self.menuFile.addAction(self.actionLoad_Data)
        self.menuPreprocessing.addAction(self.actionLoad_Preprocessing)
        self.menuPreprocessing.addAction(self.actionSpectral_Indices)
        self.menuTransform.addAction(self.actionLoad_Transform)
        self.menuRegression.addAction(self.actionCross_Validation)
        self.menuRegression.addAction(self.actionLoad_Regression)
        self.menuFeature_Selectionj.addAction(self.actionFeature_Selection)
        self.menuFeature_Selectionj.addAction(self.actionMultiple_Classes)
        self.menuDimensionality_Reduction.addAction(self.actionDimension_Reduction)
        self.menuClassification.addAction(self.actionSpectral_Distance)
        self.menuClassification.addAction(self.actionClassification)
        self.menuClassification.addAction(self.actionGrid_Search)
        self.menuVisualizer.addAction(self.actionVisualizer)
        self.menuVisualizer.addAction(self.actionView_Data)
        self.menuVisualizer.addAction(self.actionResampling)
        self.menuVisualizer.addAction(self.actionDerivative)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuPreprocessing.menuAction())
        self.menubar.addAction(self.menuTransform.menuAction())
        self.menubar.addAction(self.menuRegression.menuAction())
        self.menubar.addAction(self.menuFeature_Selectionj.menuAction())
        self.menubar.addAction(self.menuDimensionality_Reduction.menuAction())
        self.menubar.addAction(self.menuClassification.menuAction())
        self.menubar.addAction(self.menuVisualizer.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Proximal Data Analysis"))
        self.textBrowser.setToolTip(_translate("MainWindow", "<html><head/><body><p>Console window<br/>This window gives you information about your running module<br/>Errors will also show up here, if they occur</p></body></html>"))
        self.progressBar.setToolTip(_translate("MainWindow", "<html><head/><body><p>Your current progression.</p></body></html>"))
        self.progressBar.setFormat(_translate("MainWindow", "%p%"))
        self.okPushButton.setToolTip(_translate("MainWindow", "<html><head/><body><p>Re-run your last run module</p></body></html>"))
        self.okPushButton.setText(_translate("MainWindow", "Run"))
        self.stopPushButton.setToolTip(_translate("MainWindow", "<html><head/><body><p>Completely stop the currently running module</p></body></html>"))
        self.stopPushButton.setText(_translate("MainWindow", "Stop"))
        self.deletePushButton.setToolTip(_translate("MainWindow", "<html><head/><body><p>Press this button when you are ready to run the modules in your workflow</p></body></html>"))
        self.deletePushButton.setWhatsThis(_translate("MainWindow", "Press this button when you\'re ready to run. (Ctrl+Enter)"))
        self.deletePushButton.setText(_translate("MainWindow", "Delete"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuPreprocessing.setTitle(_translate("MainWindow", "Preprocessing"))
        self.menuTransform.setTitle(_translate("MainWindow", "Transform"))
        self.menuRegression.setTitle(_translate("MainWindow", "Regression"))
        self.menuFeature_Selectionj.setTitle(_translate("MainWindow", "Feature Selection"))
        self.menuDimensionality_Reduction.setTitle(_translate("MainWindow", "Dimensionality Reduction"))
        self.menuClassification.setTitle(_translate("MainWindow", "Classification"))
        self.menuVisualizer.setTitle(_translate("MainWindow", "Data "))
        self.actionLoad_Preprocessing.setText(_translate("MainWindow", "Load Preprocessing"))
        self.actionLoad_Data.setText(_translate("MainWindow", "Load Data"))
        self.actionLoad_Transform.setText(_translate("MainWindow", "Load Transform"))
        self.actionLoad_Regression.setText(_translate("MainWindow", "Regression"))
        self.actionStratified.setText(_translate("MainWindow", "Stratified Folds"))
        self.actionCross_Validation.setText(_translate("MainWindow", "Cross Validation"))
        self.actionSpectral_Indices.setText(_translate("MainWindow", "Spectral Indices"))
        self.actionFeature_Selection.setText(_translate("MainWindow", "Two Classes"))
        self.actionMultiple_Classes.setText(_translate("MainWindow", "Multiple Classes"))
        self.actionDimension_Reduction.setText(_translate("MainWindow", "Dimension Reduction"))
        self.actionSpectral_Distance.setText(_translate("MainWindow", "Spectral Distance"))
        self.actionClassification.setText(_translate("MainWindow", "Classification "))
        self.actionVisualizer.setText(_translate("MainWindow", "Visualizer"))
        self.actionView_Data.setText(_translate("MainWindow", "View Data"))
        self.actionGrid_Search.setText(_translate("MainWindow", "Grid Search"))
        self.actionResampling.setText(_translate("MainWindow", "Resampling"))
        self.actionDerivative.setText(_translate("MainWindow", "Derivative"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

