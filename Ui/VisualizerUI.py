# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\Ui\VisualizerUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1053, 551)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.label_72 = QtWidgets.QLabel(self.groupBox)
        self.label_72.setObjectName("label_72")
        self.gridLayout.addWidget(self.label_72, 2, 0, 1, 1)
        self.label_70 = QtWidgets.QLabel(self.groupBox)
        self.label_70.setObjectName("label_70")
        self.gridLayout.addWidget(self.label_70, 0, 0, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 2, 2, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 0, 2, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.label_71 = QtWidgets.QLabel(self.groupBox)
        self.label_71.setObjectName("label_71")
        self.gridLayout.addWidget(self.label_71, 1, 0, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 1, 2, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 0, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        self.tabWidget_4 = QtWidgets.QTabWidget(self.groupBox)
        self.tabWidget_4.setObjectName("tabWidget_4")
        self.tab_29 = QtWidgets.QWidget()
        self.tab_29.setObjectName("tab_29")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_29)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtWidgets.QSplitter(self.tab_29)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.groupBox_34 = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_34.setObjectName("groupBox_34")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_34)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.plotAllMuStdRb = QtWidgets.QRadioButton(self.groupBox_34)
        self.plotAllMuStdRb.setChecked(True)
        self.plotAllMuStdRb.setObjectName("plotAllMuStdRb")
        self.buttonGroup = QtWidgets.QButtonGroup(Form)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.plotAllMuStdRb)
        self.verticalLayout_2.addWidget(self.plotAllMuStdRb)
        self.plot2ClassesRb = QtWidgets.QRadioButton(self.groupBox_34)
        self.plot2ClassesRb.setObjectName("plot2ClassesRb")
        self.buttonGroup.addButton(self.plot2ClassesRb)
        self.verticalLayout_2.addWidget(self.plot2ClassesRb)
        self.plotContinuumRb = QtWidgets.QRadioButton(self.groupBox_34)
        self.plotContinuumRb.setChecked(False)
        self.plotContinuumRb.setObjectName("plotContinuumRb")
        self.buttonGroup.addButton(self.plotContinuumRb)
        self.verticalLayout_2.addWidget(self.plotContinuumRb)
        self.label_73 = QtWidgets.QLabel(self.groupBox_34)
        self.label_73.setObjectName("label_73")
        self.verticalLayout_2.addWidget(self.label_73)
        self.baselineLe = QtWidgets.QLineEdit(self.groupBox_34)
        self.baselineLe.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.baselineLe.sizePolicy().hasHeightForWidth())
        self.baselineLe.setSizePolicy(sizePolicy)
        self.baselineLe.setObjectName("baselineLe")
        self.verticalLayout_2.addWidget(self.baselineLe)
        self.label = QtWidgets.QLabel(self.groupBox_34)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.frame = QtWidgets.QFrame(self.groupBox_34)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_58 = QtWidgets.QLabel(self.frame)
        self.label_58.setEnabled(False)
        self.label_58.setObjectName("label_58")
        self.verticalLayout.addWidget(self.label_58)
        self.comboBox_25 = QtWidgets.QComboBox(self.frame)
        self.comboBox_25.setEnabled(False)
        self.comboBox_25.setObjectName("comboBox_25")
        self.verticalLayout.addWidget(self.comboBox_25)
        self.label_59 = QtWidgets.QLabel(self.frame)
        self.label_59.setEnabled(False)
        self.label_59.setObjectName("label_59")
        self.verticalLayout.addWidget(self.label_59)
        self.comboBox_26 = QtWidgets.QComboBox(self.frame)
        self.comboBox_26.setEnabled(False)
        self.comboBox_26.setObjectName("comboBox_26")
        self.verticalLayout.addWidget(self.comboBox_26)
        self.verticalLayout_2.addWidget(self.frame)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.mplWidgetSpectral_5 = QMatplotlibWidget(self.splitter)
        self.mplWidgetSpectral_5.setObjectName("mplWidgetSpectral_5")
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_29, "")
        self.tab_44 = QtWidgets.QWidget()
        self.tab_44.setObjectName("tab_44")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_44)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.splitter_2 = QtWidgets.QSplitter(self.tab_44)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.groupBox_36 = QtWidgets.QGroupBox(self.splitter_2)
        self.groupBox_36.setObjectName("groupBox_36")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox_36)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.plotDistnRb = QtWidgets.QRadioButton(self.groupBox_36)
        self.plotDistnRb.setEnabled(True)
        self.plotDistnRb.setObjectName("plotDistnRb")
        self.buttonGroup.addButton(self.plotDistnRb)
        self.verticalLayout_8.addWidget(self.plotDistnRb)
        self.label_76 = QtWidgets.QLabel(self.groupBox_36)
        self.label_76.setObjectName("label_76")
        self.verticalLayout_8.addWidget(self.label_76)
        self.wavelengthCmb = QtWidgets.QComboBox(self.groupBox_36)
        self.wavelengthCmb.setObjectName("wavelengthCmb")
        self.verticalLayout_8.addWidget(self.wavelengthCmb)
        spacerItem1 = QtWidgets.QSpacerItem(20, 263, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_8.addItem(spacerItem1)
        self.mplWidgetDist = QMatplotlibWidget(self.splitter_2)
        self.mplWidgetDist.setObjectName("mplWidgetDist")
        self.gridLayout_4.addWidget(self.splitter_2, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_44, "")
        self.tab_33 = QtWidgets.QWidget()
        self.tab_33.setObjectName("tab_33")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.tab_33)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.splitter_3 = QtWidgets.QSplitter(self.tab_33)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.groupBox_38 = QtWidgets.QGroupBox(self.splitter_3)
        self.groupBox_38.setObjectName("groupBox_38")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_38)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.sdiRb = QtWidgets.QRadioButton(self.groupBox_38)
        self.sdiRb.setObjectName("sdiRb")
        self.buttonGroup.addButton(self.sdiRb)
        self.verticalLayout_9.addWidget(self.sdiRb)
        self.annovaRb = QtWidgets.QRadioButton(self.groupBox_38)
        self.annovaRb.setObjectName("annovaRb")
        self.buttonGroup.addButton(self.annovaRb)
        self.verticalLayout_9.addWidget(self.annovaRb)
        self.tukeyRb = QtWidgets.QRadioButton(self.groupBox_38)
        self.tukeyRb.setChecked(False)
        self.tukeyRb.setObjectName("tukeyRb")
        self.buttonGroup.addButton(self.tukeyRb)
        self.verticalLayout_9.addWidget(self.tukeyRb)
        self.label_67 = QtWidgets.QLabel(self.groupBox_38)
        self.label_67.setObjectName("label_67")
        self.verticalLayout_9.addWidget(self.label_67)
        self.wavelengthCmb2 = QtWidgets.QComboBox(self.groupBox_38)
        self.wavelengthCmb2.setEnabled(False)
        self.wavelengthCmb2.setObjectName("wavelengthCmb2")
        self.verticalLayout_9.addWidget(self.wavelengthCmb2)
        self.kruskalRb = QtWidgets.QRadioButton(self.groupBox_38)
        self.kruskalRb.setObjectName("kruskalRb")
        self.buttonGroup.addButton(self.kruskalRb)
        self.verticalLayout_9.addWidget(self.kruskalRb)
        self.label_3 = QtWidgets.QLabel(self.groupBox_38)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_9.addWidget(self.label_3)
        self.frame_3 = QtWidgets.QFrame(self.groupBox_38)
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_64 = QtWidgets.QLabel(self.frame_3)
        self.label_64.setEnabled(False)
        self.label_64.setObjectName("label_64")
        self.verticalLayout_10.addWidget(self.label_64)
        self.comboBox_classA_8 = QtWidgets.QComboBox(self.frame_3)
        self.comboBox_classA_8.setEnabled(False)
        self.comboBox_classA_8.setObjectName("comboBox_classA_8")
        self.verticalLayout_10.addWidget(self.comboBox_classA_8)
        self.label_65 = QtWidgets.QLabel(self.frame_3)
        self.label_65.setEnabled(False)
        self.label_65.setObjectName("label_65")
        self.verticalLayout_10.addWidget(self.label_65)
        self.comboBox_classB_8 = QtWidgets.QComboBox(self.frame_3)
        self.comboBox_classB_8.setEnabled(False)
        self.comboBox_classB_8.setObjectName("comboBox_classB_8")
        self.verticalLayout_10.addWidget(self.comboBox_classB_8)
        self.verticalLayout_9.addWidget(self.frame_3)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem2)
        self.mplWidgetSep = QMatplotlibWidget(self.splitter_3)
        self.mplWidgetSep.setObjectName("mplWidgetSep")
        self.gridLayout_8.addWidget(self.splitter_3, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_33, "")
        self.tab_34 = QtWidgets.QWidget()
        self.tab_34.setObjectName("tab_34")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_34)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.splitter_4 = QtWidgets.QSplitter(self.tab_34)
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName("splitter_4")
        self.groupBox_40 = QtWidgets.QGroupBox(self.splitter_4)
        self.groupBox_40.setObjectName("groupBox_40")
        self.verticalLayout_93 = QtWidgets.QVBoxLayout(self.groupBox_40)
        self.verticalLayout_93.setObjectName("verticalLayout_93")
        self.plotCorrRb = QtWidgets.QRadioButton(self.groupBox_40)
        self.plotCorrRb.setChecked(False)
        self.plotCorrRb.setObjectName("plotCorrRb")
        self.buttonGroup.addButton(self.plotCorrRb)
        self.verticalLayout_93.addWidget(self.plotCorrRb)
        spacerItem3 = QtWidgets.QSpacerItem(20, 118, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_93.addItem(spacerItem3)
        self.widget_corr_5 = QMatplotlibWidget(self.splitter_4)
        self.widget_corr_5.setObjectName("widget_corr_5")
        self.gridLayout_5.addWidget(self.splitter_4, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_34, "")
        self.tab___ = QtWidgets.QWidget()
        self.tab___.setObjectName("tab___")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab___)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.splitter_5 = QtWidgets.QSplitter(self.tab___)
        self.splitter_5.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_5.setObjectName("splitter_5")
        self.groupBox_43 = QtWidgets.QGroupBox(self.splitter_5)
        self.groupBox_43.setObjectName("groupBox_43")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.groupBox_43)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.plotStatsRb = QtWidgets.QRadioButton(self.groupBox_43)
        self.plotStatsRb.setChecked(False)
        self.plotStatsRb.setObjectName("plotStatsRb")
        self.buttonGroup.addButton(self.plotStatsRb)
        self.verticalLayout_13.addWidget(self.plotStatsRb)
        self.label_74 = QtWidgets.QLabel(self.groupBox_43)
        self.label_74.setEnabled(False)
        self.label_74.setObjectName("label_74")
        self.verticalLayout_13.addWidget(self.label_74)
        self.comboBox_31 = QtWidgets.QComboBox(self.groupBox_43)
        self.comboBox_31.setEnabled(False)
        self.comboBox_31.setObjectName("comboBox_31")
        self.verticalLayout_13.addWidget(self.comboBox_31)
        spacerItem4 = QtWidgets.QSpacerItem(70, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_13.addItem(spacerItem4)
        self.widget_corr_6 = QMatplotlibWidget(self.splitter_5)
        self.widget_corr_6.setObjectName("widget_corr_6")
        self.gridLayout_6.addWidget(self.splitter_5, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab___, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.splitter_6 = QtWidgets.QSplitter(self.tab_2)
        self.splitter_6.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_6.setObjectName("splitter_6")
        self.groupBox_45 = QtWidgets.QGroupBox(self.splitter_6)
        self.groupBox_45.setObjectName("groupBox_45")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.groupBox_45)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.plotOneParamRb = QtWidgets.QRadioButton(self.groupBox_45)
        self.plotOneParamRb.setChecked(False)
        self.plotOneParamRb.setObjectName("plotOneParamRb")
        self.buttonGroup.addButton(self.plotOneParamRb)
        self.verticalLayout_15.addWidget(self.plotOneParamRb)
        self.label_77 = QtWidgets.QLabel(self.groupBox_45)
        self.label_77.setEnabled(False)
        self.label_77.setObjectName("label_77")
        self.verticalLayout_15.addWidget(self.label_77)
        self.comboBox_33 = QtWidgets.QComboBox(self.groupBox_45)
        self.comboBox_33.setEnabled(False)
        self.comboBox_33.setObjectName("comboBox_33")
        self.verticalLayout_15.addWidget(self.comboBox_33)
        self.plotAllParamRb = QtWidgets.QRadioButton(self.groupBox_45)
        self.plotAllParamRb.setObjectName("plotAllParamRb")
        self.buttonGroup.addButton(self.plotAllParamRb)
        self.verticalLayout_15.addWidget(self.plotAllParamRb)
        spacerItem5 = QtWidgets.QSpacerItem(139, 28, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_15.addItem(spacerItem5)
        self.widget_corr_7 = QMatplotlibWidget(self.splitter_6)
        self.widget_corr_7.setObjectName("widget_corr_7")
        self.gridLayout_7.addWidget(self.splitter_6, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_2, "")
        self.verticalLayout_3.addWidget(self.tabWidget_4)
        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget_4.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Timeseries Analysis"))
        self.groupBox.setTitle(_translate("Form", "Visualizer"))
        self.label_72.setText(_translate("Form", "Output"))
        self.label_70.setText(_translate("Form", "Spectra"))
        self.pushButton_6.setText(_translate("Form", "Save"))
        self.pushButton_4.setText(_translate("Form", "Browse"))
        self.label_71.setText(_translate("Form", "Metadata"))
        self.pushButton_5.setText(_translate("Form", "Browse"))
        self.groupBox_34.setTitle(_translate("Form", "Category"))
        self.plotAllMuStdRb.setText(_translate("Form", "All Spectra (Mean & Standard Deviation)"))
        self.plot2ClassesRb.setText(_translate("Form", "Compare Two Class Spectra"))
        self.plotContinuumRb.setText(_translate("Form", "Continum Removal"))
        self.label_73.setText(_translate("Form", "Baseline"))
        self.baselineLe.setText(_translate("Form", "0.93"))
        self.label.setText(_translate("Form", "Compare"))
        self.label_58.setText(_translate("Form", "Class A"))
        self.label_59.setText(_translate("Form", "Class B"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_29), _translate("Form", "Spectral View"))
        self.groupBox_36.setTitle(_translate("Form", "Category"))
        self.plotDistnRb.setText(_translate("Form", "Band Plot"))
        self.label_76.setText(_translate("Form", "Wavelength"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_44), _translate("Form", "Distribution"))
        self.groupBox_38.setTitle(_translate("Form", "Category"))
        self.sdiRb.setText(_translate("Form", "Spectral Discrimination Index"))
        self.annovaRb.setText(_translate("Form", "Oneway Anova"))
        self.tukeyRb.setText(_translate("Form", "Tukey’s multi-comparison"))
        self.label_67.setText(_translate("Form", "Wavelength"))
        self.kruskalRb.setText(_translate("Form", "Kruskal Wallis Non-Parametric H Test"))
        self.label_3.setText(_translate("Form", "Compare :"))
        self.label_64.setText(_translate("Form", "Class A"))
        self.label_65.setText(_translate("Form", "Class B"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_33), _translate("Form", "Band Seperability"))
        self.groupBox_40.setTitle(_translate("Form", "Category"))
        self.plotCorrRb.setText(_translate("Form", "Band Correlation"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_34), _translate("Form", "Band Correlation"))
        self.groupBox_43.setTitle(_translate("Form", "Category"))
        self.plotStatsRb.setText(_translate("Form", "Statistics"))
        self.label_74.setText(_translate("Form", "Parameter"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab___), _translate("Form", "Class Vs Metadata"))
        self.groupBox_45.setTitle(_translate("Form", "Category"))
        self.plotOneParamRb.setText(_translate("Form", "Single parameter"))
        self.label_77.setText(_translate("Form", "Parameter"))
        self.plotAllParamRb.setText(_translate("Form", "All Parameter"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_2), _translate("Form", "Metadata_Statistics"))

from Ui.qmatplotlibwidget import QMatplotlibWidget