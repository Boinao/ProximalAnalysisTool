# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialogROIseparability.Ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_HxToolsDialog(object):
    def setupUi(self, HxToolsDialog):
        HxToolsDialog.setObjectName("HxToolsDialog")
        HxToolsDialog.resize(1020, 703)
        self.gridLayout_3 = QtWidgets.QGridLayout(HxToolsDialog)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(HxToolsDialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(HxToolsDialog)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(HxToolsDialog)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(HxToolsDialog)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 0, 1, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(HxToolsDialog)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(HxToolsDialog)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout, 0, 0, 3, 1)
        self.pushButton_4 = QtWidgets.QPushButton(HxToolsDialog)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_3.addWidget(self.pushButton_4, 0, 1, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(HxToolsDialog)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout_3.addWidget(self.pushButton_5, 1, 1, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(HxToolsDialog)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout_3.addWidget(self.pushButton_6, 2, 1, 1, 1)
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.tabWidget = QtWidgets.QTabWidget(HxToolsDialog)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.rb_sep_all = QtWidgets.QRadioButton(self.groupBox_3)
        self.rb_sep_all.setChecked(True)
        self.rb_sep_all.setObjectName("rb_sep_all")
        self.buttonGroup = QtWidgets.QButtonGroup(HxToolsDialog)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.rb_sep_all)
        self.verticalLayout.addWidget(self.rb_sep_all)
        self.rb_sep_cv = QtWidgets.QRadioButton(self.groupBox_3)
        self.rb_sep_cv.setObjectName("rb_sep_cv")
        self.buttonGroup.addButton(self.rb_sep_cv)
        self.verticalLayout.addWidget(self.rb_sep_cv)
        self.rb_sep_compare = QtWidgets.QRadioButton(self.groupBox_3)
        self.rb_sep_compare.setObjectName("rb_sep_compare")
        self.buttonGroup.addButton(self.rb_sep_compare)
        self.verticalLayout.addWidget(self.rb_sep_compare)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setEnabled(False)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox.setEnabled(False)
        self.comboBox.setMaxVisibleItems(10)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setEnabled(False)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_6.addWidget(self.label_14)
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_2.setEnabled(False)
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_6.addWidget(self.comboBox_2)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        spacerItem = QtWidgets.QSpacerItem(20, 178, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_8.addLayout(self.verticalLayout_2)
        self.mplWidgetSpectral = QMatplotlibWidget(self.tab)
        self.mplWidgetSpectral.setObjectName("mplWidgetSpectral")
        self.horizontalLayout_8.addWidget(self.mplWidgetSpectral)
        self.gridLayout_15.addLayout(self.horizontalLayout_8, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.radioClassDistribute = QtWidgets.QRadioButton(self.groupBox_5)
        self.radioClassDistribute.setChecked(False)
        self.radioClassDistribute.setObjectName("radioClassDistribute")
        self.buttonGroup.addButton(self.radioClassDistribute)
        self.verticalLayout_9.addWidget(self.radioClassDistribute)
        self.radioFeatureDistribute = QtWidgets.QRadioButton(self.groupBox_5)
        self.radioFeatureDistribute.setEnabled(True)
        self.radioFeatureDistribute.setObjectName("radioFeatureDistribute")
        self.buttonGroup.addButton(self.radioFeatureDistribute)
        self.verticalLayout_9.addWidget(self.radioFeatureDistribute)
        self.radioFeatureVar = QtWidgets.QRadioButton(self.groupBox_5)
        self.radioFeatureVar.setEnabled(True)
        self.radioFeatureVar.setObjectName("radioFeatureVar")
        self.buttonGroup.addButton(self.radioFeatureVar)
        self.verticalLayout_9.addWidget(self.radioFeatureVar)
        self.label_4 = QtWidgets.QLabel(self.groupBox_5)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_9.addWidget(self.label_4)
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox_5)
        self.comboBox_3.setEnabled(False)
        self.comboBox_3.setObjectName("comboBox_3")
        self.verticalLayout_9.addWidget(self.comboBox_3)
        self.gridLayout_5.addLayout(self.verticalLayout_9, 0, 0, 1, 1)
        self.verticalLayout_10.addWidget(self.groupBox_5)
        spacerItem1 = QtWidgets.QSpacerItem(138, 248, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_10.addItem(spacerItem1)
        self.gridLayout_7.addLayout(self.verticalLayout_10, 0, 0, 1, 1)
        self.widget_dist = QMatplotlibWidget(self.tab_2)
        self.widget_dist.setObjectName("widget_dist")
        self.gridLayout_7.addWidget(self.widget_dist, 0, 1, 1, 1)
        self.gridLayout_14.addLayout(self.gridLayout_7, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.radioButton_PCA = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_PCA.setEnabled(True)
        self.radioButton_PCA.setChecked(False)
        self.radioButton_PCA.setObjectName("radioButton_PCA")
        self.buttonGroup.addButton(self.radioButton_PCA)
        self.verticalLayout_14.addWidget(self.radioButton_PCA)
        self.radioButton_TSNE = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_TSNE.setChecked(False)
        self.radioButton_TSNE.setObjectName("radioButton_TSNE")
        self.buttonGroup.addButton(self.radioButton_TSNE)
        self.verticalLayout_14.addWidget(self.radioButton_TSNE)
        self.label_8 = QtWidgets.QLabel(self.groupBox_4)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_14.addWidget(self.label_8)
        self.x1CoordEdit = QtWidgets.QLineEdit(self.groupBox_4)
        self.x1CoordEdit.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x1CoordEdit.sizePolicy().hasHeightForWidth())
        self.x1CoordEdit.setSizePolicy(sizePolicy)
        self.x1CoordEdit.setMaxLength(3)
        self.x1CoordEdit.setObjectName("x1CoordEdit")
        self.verticalLayout_14.addWidget(self.x1CoordEdit)
        self.gridLayout_2.addLayout(self.verticalLayout_14, 0, 0, 1, 1)
        self.verticalLayout_13.addWidget(self.groupBox_4)
        spacerItem2 = QtWidgets.QSpacerItem(13, 288, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_13.addItem(spacerItem2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_13)
        self.widget_3 = QMatplotlibWidget(self.tab_3)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_4.addWidget(self.widget_3)
        self.gridLayout_10.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.gridLayout_12 = QtWidgets.QGridLayout()
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.radioFRA = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioFRA.setChecked(False)
        self.radioFRA.setObjectName("radioFRA")
        self.buttonGroup.addButton(self.radioFRA)
        self.verticalLayout_8.addWidget(self.radioFRA)
        self.radioSAM = QtWidgets.QRadioButton(self.groupBox_6)
        self.radioSAM.setChecked(False)
        self.radioSAM.setObjectName("radioSAM")
        self.buttonGroup.addButton(self.radioSAM)
        self.verticalLayout_8.addWidget(self.radioSAM)
        self.label_11 = QtWidgets.QLabel(self.groupBox_6)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_8.addWidget(self.label_11)
        self.comboBox_8 = QtWidgets.QComboBox(self.groupBox_6)
        self.comboBox_8.setEnabled(False)
        self.comboBox_8.setObjectName("comboBox_8")
        self.verticalLayout_8.addWidget(self.comboBox_8)
        self.verticalLayout_11.addLayout(self.verticalLayout_8)
        spacerItem3 = QtWidgets.QSpacerItem(20, 158, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem3)
        self.gridLayout_9.addLayout(self.verticalLayout_11, 0, 0, 1, 1)
        self.gridLayout_12.addWidget(self.groupBox_6, 0, 0, 1, 1)
        self.horizontalLayout_7.addLayout(self.gridLayout_12)
        self.widget_class_sep = QMatplotlibWidget(self.tab_4)
        self.widget_class_sep.setObjectName("widget_class_sep")
        self.horizontalLayout_7.addWidget(self.widget_class_sep)
        self.gridLayout_4.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.tab_5)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout()
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.groupBox = QtWidgets.QGroupBox(self.tab_5)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.radioFTEST = QtWidgets.QRadioButton(self.groupBox)
        self.radioFTEST.setChecked(False)
        self.radioFTEST.setObjectName("radioFTEST")
        self.buttonGroup.addButton(self.radioFTEST)
        self.verticalLayout_5.addWidget(self.radioFTEST)
        self.radioSEP = QtWidgets.QRadioButton(self.groupBox)
        self.radioSEP.setObjectName("radioSEP")
        self.buttonGroup.addButton(self.radioSEP)
        self.verticalLayout_5.addWidget(self.radioSEP)
        self.radioMUTUAL = QtWidgets.QRadioButton(self.groupBox)
        self.radioMUTUAL.setObjectName("radioMUTUAL")
        self.buttonGroup.addButton(self.radioMUTUAL)
        self.verticalLayout_5.addWidget(self.radioMUTUAL)
        self.radioTREE = QtWidgets.QRadioButton(self.groupBox)
        self.radioTREE.setObjectName("radioTREE")
        self.buttonGroup.addButton(self.radioTREE)
        self.verticalLayout_5.addWidget(self.radioTREE)
        self.gridLayout_11.addLayout(self.verticalLayout_5, 0, 0, 1, 1)
        self.verticalLayout_17.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_5)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_19 = QtWidgets.QLabel(self.groupBox_2)
        self.label_19.setEnabled(False)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_3.addWidget(self.label_19)
        self.comboBox_classA = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_classA.setEnabled(False)
        self.comboBox_classA.setObjectName("comboBox_classA")
        self.horizontalLayout_3.addWidget(self.comboBox_classA)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        self.label_18.setEnabled(False)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_14.addWidget(self.label_18)
        self.comboBox_classB = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_classB.setEnabled(False)
        self.comboBox_classB.setObjectName("comboBox_classB")
        self.horizontalLayout_14.addWidget(self.comboBox_classB)
        self.verticalLayout_4.addLayout(self.horizontalLayout_14)
        self.verticalLayout_12.addLayout(self.verticalLayout_4)
        self.verticalLayout_17.addWidget(self.groupBox_2)
        self.verticalLayout_18.addLayout(self.verticalLayout_17)
        spacerItem4 = QtWidgets.QSpacerItem(20, 178, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_18.addItem(spacerItem4)
        self.horizontalLayout_15.addLayout(self.verticalLayout_18)
        self.widget_feature = QMatplotlibWidget(self.tab_5)
        self.widget_feature.setObjectName("widget_feature")
        self.horizontalLayout_15.addWidget(self.widget_feature)
        self.gridLayout_8.addLayout(self.horizontalLayout_15, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab_6)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.groupBox_7 = QtWidgets.QGroupBox(self.tab_6)
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.radio_classcorr = QtWidgets.QRadioButton(self.groupBox_7)
        self.radio_classcorr.setChecked(False)
        self.radio_classcorr.setObjectName("radio_classcorr")
        self.buttonGroup.addButton(self.radio_classcorr)
        self.verticalLayout_6.addWidget(self.radio_classcorr)
        self.radioPairplot = QtWidgets.QRadioButton(self.groupBox_7)
        self.radioPairplot.setObjectName("radioPairplot")
        self.buttonGroup.addButton(self.radioPairplot)
        self.verticalLayout_6.addWidget(self.radioPairplot)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_17 = QtWidgets.QLabel(self.groupBox_7)
        self.label_17.setEnabled(False)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_11.addWidget(self.label_17)
        self.comboBox_6 = QtWidgets.QComboBox(self.groupBox_7)
        self.comboBox_6.setEnabled(False)
        self.comboBox_6.setObjectName("comboBox_6")
        self.horizontalLayout_11.addWidget(self.comboBox_6)
        self.verticalLayout_6.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_16 = QtWidgets.QLabel(self.groupBox_7)
        self.label_16.setEnabled(False)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_12.addWidget(self.label_16)
        self.comboBox_7 = QtWidgets.QComboBox(self.groupBox_7)
        self.comboBox_7.setEnabled(False)
        self.comboBox_7.setObjectName("comboBox_7")
        self.horizontalLayout_12.addWidget(self.comboBox_7)
        self.verticalLayout_6.addLayout(self.horizontalLayout_12)
        self.verticalLayout_16.addWidget(self.groupBox_7)
        spacerItem5 = QtWidgets.QSpacerItem(20, 278, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_16.addItem(spacerItem5)
        self.horizontalLayout_13.addLayout(self.verticalLayout_16)
        self.widget_corr = QMatplotlibWidget(self.tab_6)
        self.widget_corr.setObjectName("widget_corr")
        self.horizontalLayout_13.addWidget(self.widget_corr)
        self.gridLayout_6.addLayout(self.horizontalLayout_13, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_6, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.tab_7)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.groupBox_8 = QtWidgets.QGroupBox(self.tab_7)
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_8)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.radioButton_bandwise = QtWidgets.QRadioButton(self.groupBox_8)
        self.radioButton_bandwise.setObjectName("radioButton_bandwise")
        self.buttonGroup.addButton(self.radioButton_bandwise)
        self.verticalLayout_3.addWidget(self.radioButton_bandwise)
        self.radioButton_bandnorm = QtWidgets.QRadioButton(self.groupBox_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radioButton_bandnorm.sizePolicy().hasHeightForWidth())
        self.radioButton_bandnorm.setSizePolicy(sizePolicy)
        self.radioButton_bandnorm.setObjectName("radioButton_bandnorm")
        self.buttonGroup.addButton(self.radioButton_bandnorm)
        self.verticalLayout_3.addWidget(self.radioButton_bandnorm)
        self.label_13 = QtWidgets.QLabel(self.groupBox_8)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_3.addWidget(self.label_13)
        self.x1CoordEdit_2 = QtWidgets.QLineEdit(self.groupBox_8)
        self.x1CoordEdit_2.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x1CoordEdit_2.sizePolicy().hasHeightForWidth())
        self.x1CoordEdit_2.setSizePolicy(sizePolicy)
        self.x1CoordEdit_2.setObjectName("x1CoordEdit_2")
        self.verticalLayout_3.addWidget(self.x1CoordEdit_2)
        self.verticalLayout_7.addLayout(self.verticalLayout_3)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_12 = QtWidgets.QLabel(self.groupBox_8)
        self.label_12.setEnabled(False)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_5.addWidget(self.label_12)
        self.comboBox_classA_1 = QtWidgets.QComboBox(self.groupBox_8)
        self.comboBox_classA_1.setEnabled(False)
        self.comboBox_classA_1.setObjectName("comboBox_classA_1")
        self.horizontalLayout_5.addWidget(self.comboBox_classA_1)
        self.verticalLayout_7.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_15 = QtWidgets.QLabel(self.groupBox_8)
        self.label_15.setEnabled(False)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_9.addWidget(self.label_15)
        self.comboBox_classB_1 = QtWidgets.QComboBox(self.groupBox_8)
        self.comboBox_classB_1.setEnabled(False)
        self.comboBox_classB_1.setObjectName("comboBox_classB_1")
        self.horizontalLayout_9.addWidget(self.comboBox_classB_1)
        self.verticalLayout_7.addLayout(self.horizontalLayout_9)
        self.verticalLayout_15.addWidget(self.groupBox_8)
        spacerItem6 = QtWidgets.QSpacerItem(128, 238, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_15.addItem(spacerItem6)
        self.horizontalLayout_10.addLayout(self.verticalLayout_15)
        self.widget_band_sep = QMatplotlibWidget(self.tab_7)
        self.widget_band_sep.setObjectName("widget_band_sep")
        self.horizontalLayout_10.addWidget(self.widget_band_sep)
        self.gridLayout_13.addLayout(self.horizontalLayout_10, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_7, "")
        self.verticalLayout_19.addWidget(self.tabWidget)
        self.gridLayout_3.addLayout(self.verticalLayout_19, 3, 0, 1, 1)

        self.retranslateUi(HxToolsDialog)
        self.tabWidget.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(HxToolsDialog)

    def retranslateUi(self, HxToolsDialog):
        _translate = QtCore.QCoreApplication.translate
        HxToolsDialog.setWindowTitle(_translate("HxToolsDialog", "ROI Seperability Analysis"))
        self.label.setText(_translate("HxToolsDialog", "Input Multi-Channel Image"))
        self.label_6.setText(_translate("HxToolsDialog", "Ground Truth Raster "))
        self.label_10.setText(_translate("HxToolsDialog", "Output Statistics File"))
        self.pushButton_4.setText(_translate("HxToolsDialog", "Browse"))
        self.pushButton_5.setText(_translate("HxToolsDialog", "Browse"))
        self.pushButton_6.setText(_translate("HxToolsDialog", "Browse"))
        self.groupBox_3.setTitle(_translate("HxToolsDialog", "Category"))
        self.rb_sep_all.setText(_translate("HxToolsDialog", "All Spectra (Mean)"))
        self.rb_sep_cv.setText(_translate("HxToolsDialog", "All Spectra (Coefficient of Variation)"))
        self.rb_sep_compare.setText(_translate("HxToolsDialog", "Compare Two Class Spectra"))
        self.label_5.setText(_translate("HxToolsDialog", "Class A"))
        self.label_14.setText(_translate("HxToolsDialog", "Class B"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("HxToolsDialog", "Spectral View"))
        self.groupBox_5.setTitle(_translate("HxToolsDialog", "Category"))
        self.radioClassDistribute.setText(_translate("HxToolsDialog", "Class Distribution"))
        self.radioFeatureDistribute.setText(_translate("HxToolsDialog", "Feature Distribution"))
        self.radioFeatureVar.setText(_translate("HxToolsDialog", "Feature Variability"))
        self.label_4.setText(_translate("HxToolsDialog", "Select Band"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("HxToolsDialog", "Distribution"))
        self.groupBox_4.setTitle(_translate("HxToolsDialog", "Transform"))
        self.radioButton_PCA.setText(_translate("HxToolsDialog", "PCA"))
        self.radioButton_TSNE.setText(_translate("HxToolsDialog", "t-SNE"))
        self.label_8.setText(_translate("HxToolsDialog", "Perplexity"))
        self.x1CoordEdit.setText(_translate("HxToolsDialog", "30"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("HxToolsDialog", "Low-Dimension Projection"))
        self.groupBox_6.setTitle(_translate("HxToolsDialog", "Method"))
        self.radioFRA.setText(_translate("HxToolsDialog", "Fractional Distance"))
        self.radioSAM.setText(_translate("HxToolsDialog", "SAM"))
        self.label_11.setText(_translate("HxToolsDialog", "Select the Class for Comparison"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("HxToolsDialog", "Class Separability"))
        self.groupBox.setTitle(_translate("HxToolsDialog", "Methods"))
        self.radioFTEST.setText(_translate("HxToolsDialog", "F-test"))
        self.radioSEP.setText(_translate("HxToolsDialog", "Spectral Seperability"))
        self.radioMUTUAL.setText(_translate("HxToolsDialog", "Mutual Information"))
        self.radioTREE.setText(_translate("HxToolsDialog", "Trees Ensemble"))
        self.groupBox_2.setTitle(_translate("HxToolsDialog", "Compare"))
        self.label_19.setText(_translate("HxToolsDialog", "Class A"))
        self.label_18.setText(_translate("HxToolsDialog", "Class B"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("HxToolsDialog", "Feature Identification"))
        self.groupBox_7.setTitle(_translate("HxToolsDialog", "Category"))
        self.radio_classcorr.setText(_translate("HxToolsDialog", "Band Correlation"))
        self.radioPairplot.setText(_translate("HxToolsDialog", "Band Pair Plot"))
        self.label_17.setText(_translate("HxToolsDialog", "Band A"))
        self.label_16.setText(_translate("HxToolsDialog", "Band B"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("HxToolsDialog", "Band Correlation"))
        self.groupBox_8.setTitle(_translate("HxToolsDialog", "Compare"))
        self.radioButton_bandwise.setText(_translate("HxToolsDialog", "Band-wise"))
        self.radioButton_bandnorm.setText(_translate("HxToolsDialog", "[A-B]/[A+B]"))
        self.label_13.setText(_translate("HxToolsDialog", "Band-width"))
        self.x1CoordEdit_2.setText(_translate("HxToolsDialog", "1"))
        self.label_12.setText(_translate("HxToolsDialog", "Class A"))
        self.label_15.setText(_translate("HxToolsDialog", "Class B"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("HxToolsDialog", "Spectral Descrimination Matrix"))

from HxTools.ui.qmatplotlibwidget import QMatplotlibWidget
