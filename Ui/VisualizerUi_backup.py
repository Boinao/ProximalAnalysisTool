# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'VisualizerUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1053, 595)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 0, 1, 1, 1)
        self.label_71 = QtWidgets.QLabel(self.groupBox)
        self.label_71.setObjectName("label_71")
        self.gridLayout.addWidget(self.label_71, 1, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 0, 2, 1, 1)
        self.label_70 = QtWidgets.QLabel(self.groupBox)
        self.label_70.setObjectName("label_70")
        self.gridLayout.addWidget(self.label_70, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 1, 2, 1, 1)
        self.label_72 = QtWidgets.QLabel(self.groupBox)
        self.label_72.setObjectName("label_72")
        self.gridLayout.addWidget(self.label_72, 2, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 2, 2, 1, 1)
        self.tabWidget_4 = QtWidgets.QTabWidget(self.groupBox)
        self.tabWidget_4.setObjectName("tabWidget_4")
        self.tab_29 = QtWidgets.QWidget()
        self.tab_29.setObjectName("tab_29")
        self.gridLayout_57 = QtWidgets.QGridLayout(self.tab_29)
        self.gridLayout_57.setObjectName("gridLayout_57")
        self.horizontalLayout_61 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_61.setObjectName("horizontalLayout_61")
        self.verticalLayout_79 = QtWidgets.QVBoxLayout()
        self.verticalLayout_79.setObjectName("verticalLayout_79")
        self.groupBox_34 = QtWidgets.QGroupBox(self.tab_29)
        self.groupBox_34.setObjectName("groupBox_34")
        self.verticalLayout_80 = QtWidgets.QVBoxLayout(self.groupBox_34)
        self.verticalLayout_80.setObjectName("verticalLayout_80")
        self.rb_sep_all_5 = QtWidgets.QRadioButton(self.groupBox_34)
        self.rb_sep_all_5.setChecked(True)
        self.rb_sep_all_5.setObjectName("rb_sep_all_5")
        self.buttonGroup = QtWidgets.QButtonGroup(Form)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.rb_sep_all_5)
        self.verticalLayout_80.addWidget(self.rb_sep_all_5)
        self.rb_sep_compare_5 = QtWidgets.QRadioButton(self.groupBox_34)
        self.rb_sep_compare_5.setObjectName("rb_sep_compare_5")
        self.buttonGroup.addButton(self.rb_sep_compare_5)
        self.verticalLayout_80.addWidget(self.rb_sep_compare_5)
        self.radioClassDistribute = QtWidgets.QRadioButton(self.groupBox_34)
        self.radioClassDistribute.setChecked(False)
        self.radioClassDistribute.setObjectName("radioClassDistribute")
        self.buttonGroup.addButton(self.radioClassDistribute)
        self.verticalLayout_80.addWidget(self.radioClassDistribute)
        self.label_73 = QtWidgets.QLabel(self.groupBox_34)
        self.label_73.setObjectName("label_73")
        self.verticalLayout_80.addWidget(self.label_73)
        self.x1CoordEdit_11 = QtWidgets.QLineEdit(self.groupBox_34)
        self.x1CoordEdit_11.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x1CoordEdit_11.sizePolicy().hasHeightForWidth())
        self.x1CoordEdit_11.setSizePolicy(sizePolicy)
        self.x1CoordEdit_11.setObjectName("x1CoordEdit_11")
        self.verticalLayout_80.addWidget(self.x1CoordEdit_11)
        self.horizontalLayout_62 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_62.setObjectName("horizontalLayout_62")
        self.verticalLayout_80.addLayout(self.horizontalLayout_62)
        self.groupBox_42 = QtWidgets.QGroupBox(self.groupBox_34)
        self.groupBox_42.setObjectName("groupBox_42")
        self.verticalLayout_99 = QtWidgets.QVBoxLayout(self.groupBox_42)
        self.verticalLayout_99.setObjectName("verticalLayout_99")
        self.verticalLayout_100 = QtWidgets.QVBoxLayout()
        self.verticalLayout_100.setObjectName("verticalLayout_100")
        self.horizontalLayout_78 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_78.setObjectName("horizontalLayout_78")
        self.label_58 = QtWidgets.QLabel(self.groupBox_42)
        self.label_58.setEnabled(False)
        self.label_58.setAlignment(QtCore.Qt.AlignCenter)
        self.label_58.setObjectName("label_58")
        self.horizontalLayout_78.addWidget(self.label_58)
        self.comboBox_25 = QtWidgets.QComboBox(self.groupBox_42)
        self.comboBox_25.setEnabled(False)
        self.comboBox_25.setObjectName("comboBox_25")
        self.horizontalLayout_78.addWidget(self.comboBox_25)
        self.verticalLayout_100.addLayout(self.horizontalLayout_78)
        self.horizontalLayout_79 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_79.setObjectName("horizontalLayout_79")
        self.label_59 = QtWidgets.QLabel(self.groupBox_42)
        self.label_59.setEnabled(False)
        self.label_59.setAlignment(QtCore.Qt.AlignCenter)
        self.label_59.setObjectName("label_59")
        self.horizontalLayout_79.addWidget(self.label_59)
        self.comboBox_26 = QtWidgets.QComboBox(self.groupBox_42)
        self.comboBox_26.setEnabled(False)
        self.comboBox_26.setObjectName("comboBox_26")
        self.horizontalLayout_79.addWidget(self.comboBox_26)
        self.verticalLayout_100.addLayout(self.horizontalLayout_79)
        self.verticalLayout_99.addLayout(self.verticalLayout_100)
        self.verticalLayout_80.addWidget(self.groupBox_42)
        self.horizontalLayout_63 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_63.setObjectName("horizontalLayout_63")
        self.verticalLayout_80.addLayout(self.horizontalLayout_63)
        self.verticalLayout_79.addWidget(self.groupBox_34)
        spacerItem = QtWidgets.QSpacerItem(20, 178, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_79.addItem(spacerItem)
        self.horizontalLayout_61.addLayout(self.verticalLayout_79)
        self.mplWidgetSpectral_5 = QMatplotlibWidget(self.tab_29)
        self.mplWidgetSpectral_5.setObjectName("mplWidgetSpectral_5")
        self.horizontalLayout_61.addWidget(self.mplWidgetSpectral_5)
        self.gridLayout_57.addLayout(self.horizontalLayout_61, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_29, "")
        self.tab_44 = QtWidgets.QWidget()
        self.tab_44.setObjectName("tab_44")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_44)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_64 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_64.setObjectName("horizontalLayout_64")
        self.verticalLayout_81 = QtWidgets.QVBoxLayout()
        self.verticalLayout_81.setObjectName("verticalLayout_81")
        self.groupBox_35 = QtWidgets.QGroupBox(self.tab_44)
        self.groupBox_35.setObjectName("groupBox_35")
        self.verticalLayout_82 = QtWidgets.QVBoxLayout(self.groupBox_35)
        self.verticalLayout_82.setObjectName("verticalLayout_82")
        self.radioFeatureVar = QtWidgets.QRadioButton(self.groupBox_35)
        self.radioFeatureVar.setEnabled(True)
        self.radioFeatureVar.setObjectName("radioFeatureVar")
        self.buttonGroup.addButton(self.radioFeatureVar)
        self.verticalLayout_82.addWidget(self.radioFeatureVar)
        self.label_75 = QtWidgets.QLabel(self.groupBox_35)
        self.label_75.setObjectName("label_75")
        self.verticalLayout_82.addWidget(self.label_75)
        self.x1CoordEdit_13 = QtWidgets.QLineEdit(self.groupBox_35)
        self.x1CoordEdit_13.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x1CoordEdit_13.sizePolicy().hasHeightForWidth())
        self.x1CoordEdit_13.setSizePolicy(sizePolicy)
        self.x1CoordEdit_13.setObjectName("x1CoordEdit_13")
        self.verticalLayout_82.addWidget(self.x1CoordEdit_13)
        self.horizontalLayout_75 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_75.setObjectName("horizontalLayout_75")
        self.verticalLayout_82.addLayout(self.horizontalLayout_75)
        self.verticalLayout_81.addWidget(self.groupBox_35)
        spacerItem1 = QtWidgets.QSpacerItem(20, 178, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_81.addItem(spacerItem1)
        self.horizontalLayout_64.addLayout(self.verticalLayout_81)
        self.mplWidgetSpectral_6 = QMatplotlibWidget(self.tab_44)
        self.mplWidgetSpectral_6.setObjectName("mplWidgetSpectral_6")
        self.horizontalLayout_64.addWidget(self.mplWidgetSpectral_6)
        self.gridLayout_2.addLayout(self.horizontalLayout_64, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_44, "")
        self.tab_33 = QtWidgets.QWidget()
        self.tab_33.setObjectName("tab_33")
        self.gridLayout_66 = QtWidgets.QGridLayout(self.tab_33)
        self.gridLayout_66.setObjectName("gridLayout_66")
        self.horizontalLayout_66 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_66.setObjectName("horizontalLayout_66")
        self.verticalLayout_87 = QtWidgets.QVBoxLayout()
        self.verticalLayout_87.setObjectName("verticalLayout_87")
        self.verticalLayout_88 = QtWidgets.QVBoxLayout()
        self.verticalLayout_88.setObjectName("verticalLayout_88")
        self.groupBox_38 = QtWidgets.QGroupBox(self.tab_33)
        self.groupBox_38.setObjectName("groupBox_38")
        self.gridLayout_67 = QtWidgets.QGridLayout(self.groupBox_38)
        self.gridLayout_67.setObjectName("gridLayout_67")
        self.verticalLayout_89 = QtWidgets.QVBoxLayout()
        self.verticalLayout_89.setObjectName("verticalLayout_89")
        self.radioFTEST_5 = QtWidgets.QRadioButton(self.groupBox_38)
        self.radioFTEST_5.setChecked(False)
        self.radioFTEST_5.setObjectName("radioFTEST_5")
        self.buttonGroup.addButton(self.radioFTEST_5)
        self.verticalLayout_89.addWidget(self.radioFTEST_5)
        self.radioSEP_5 = QtWidgets.QRadioButton(self.groupBox_38)
        self.radioSEP_5.setObjectName("radioSEP_5")
        self.buttonGroup.addButton(self.radioSEP_5)
        self.verticalLayout_89.addWidget(self.radioSEP_5)
        self.radioMUTUAL_5 = QtWidgets.QRadioButton(self.groupBox_38)
        self.radioMUTUAL_5.setObjectName("radioMUTUAL_5")
        self.buttonGroup.addButton(self.radioMUTUAL_5)
        self.verticalLayout_89.addWidget(self.radioMUTUAL_5)
        self.label_67 = QtWidgets.QLabel(self.groupBox_38)
        self.label_67.setObjectName("label_67")
        self.verticalLayout_89.addWidget(self.label_67)
        self.x1CoordEdit_10 = QtWidgets.QLineEdit(self.groupBox_38)
        self.x1CoordEdit_10.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x1CoordEdit_10.sizePolicy().hasHeightForWidth())
        self.x1CoordEdit_10.setSizePolicy(sizePolicy)
        self.x1CoordEdit_10.setObjectName("x1CoordEdit_10")
        self.verticalLayout_89.addWidget(self.x1CoordEdit_10)
        self.radioTREE_5 = QtWidgets.QRadioButton(self.groupBox_38)
        self.radioTREE_5.setObjectName("radioTREE_5")
        self.buttonGroup.addButton(self.radioTREE_5)
        self.verticalLayout_89.addWidget(self.radioTREE_5)
        self.gridLayout_67.addLayout(self.verticalLayout_89, 0, 0, 1, 1)
        self.verticalLayout_88.addWidget(self.groupBox_38)
        self.groupBox_39 = QtWidgets.QGroupBox(self.tab_33)
        self.groupBox_39.setObjectName("groupBox_39")
        self.verticalLayout_90 = QtWidgets.QVBoxLayout(self.groupBox_39)
        self.verticalLayout_90.setObjectName("verticalLayout_90")
        self.verticalLayout_91 = QtWidgets.QVBoxLayout()
        self.verticalLayout_91.setObjectName("verticalLayout_91")
        self.horizontalLayout_67 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_67.setObjectName("horizontalLayout_67")
        self.label_63 = QtWidgets.QLabel(self.groupBox_39)
        self.label_63.setEnabled(False)
        self.label_63.setAlignment(QtCore.Qt.AlignCenter)
        self.label_63.setObjectName("label_63")
        self.horizontalLayout_67.addWidget(self.label_63)
        self.comboBox_classA_8 = QtWidgets.QComboBox(self.groupBox_39)
        self.comboBox_classA_8.setEnabled(False)
        self.comboBox_classA_8.setObjectName("comboBox_classA_8")
        self.horizontalLayout_67.addWidget(self.comboBox_classA_8)
        self.verticalLayout_91.addLayout(self.horizontalLayout_67)
        self.horizontalLayout_68 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_68.setObjectName("horizontalLayout_68")
        self.label_64 = QtWidgets.QLabel(self.groupBox_39)
        self.label_64.setEnabled(False)
        self.label_64.setAlignment(QtCore.Qt.AlignCenter)
        self.label_64.setObjectName("label_64")
        self.horizontalLayout_68.addWidget(self.label_64)
        self.comboBox_classB_8 = QtWidgets.QComboBox(self.groupBox_39)
        self.comboBox_classB_8.setEnabled(False)
        self.comboBox_classB_8.setObjectName("comboBox_classB_8")
        self.horizontalLayout_68.addWidget(self.comboBox_classB_8)
        self.verticalLayout_91.addLayout(self.horizontalLayout_68)
        self.verticalLayout_90.addLayout(self.verticalLayout_91)
        self.verticalLayout_88.addWidget(self.groupBox_39)
        self.verticalLayout_87.addLayout(self.verticalLayout_88)
        spacerItem2 = QtWidgets.QSpacerItem(20, 178, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_87.addItem(spacerItem2)
        self.horizontalLayout_66.addLayout(self.verticalLayout_87)
        self.widget_feature_5 = QMatplotlibWidget(self.tab_33)
        self.widget_feature_5.setObjectName("widget_feature_5")
        self.horizontalLayout_66.addWidget(self.widget_feature_5)
        self.gridLayout_66.addLayout(self.horizontalLayout_66, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_33, "")
        self.tab_34 = QtWidgets.QWidget()
        self.tab_34.setObjectName("tab_34")
        self.gridLayout_68 = QtWidgets.QGridLayout(self.tab_34)
        self.gridLayout_68.setObjectName("gridLayout_68")
        self.horizontalLayout_69 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_69.setObjectName("horizontalLayout_69")
        self.verticalLayout_92 = QtWidgets.QVBoxLayout()
        self.verticalLayout_92.setObjectName("verticalLayout_92")
        self.groupBox_40 = QtWidgets.QGroupBox(self.tab_34)
        self.groupBox_40.setObjectName("groupBox_40")
        self.verticalLayout_93 = QtWidgets.QVBoxLayout(self.groupBox_40)
        self.verticalLayout_93.setObjectName("verticalLayout_93")
        self.radio_classcorr_5 = QtWidgets.QRadioButton(self.groupBox_40)
        self.radio_classcorr_5.setChecked(False)
        self.radio_classcorr_5.setObjectName("radio_classcorr_5")
        self.buttonGroup.addButton(self.radio_classcorr_5)
        self.verticalLayout_93.addWidget(self.radio_classcorr_5)
        self.horizontalLayout_70 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_70.setObjectName("horizontalLayout_70")
        self.verticalLayout_93.addLayout(self.horizontalLayout_70)
        self.horizontalLayout_71 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_71.setObjectName("horizontalLayout_71")
        self.verticalLayout_93.addLayout(self.horizontalLayout_71)
        self.verticalLayout_92.addWidget(self.groupBox_40)
        spacerItem3 = QtWidgets.QSpacerItem(20, 278, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_92.addItem(spacerItem3)
        self.horizontalLayout_69.addLayout(self.verticalLayout_92)
        self.widget_corr_5 = QMatplotlibWidget(self.tab_34)
        self.widget_corr_5.setObjectName("widget_corr_5")
        self.horizontalLayout_69.addWidget(self.widget_corr_5)
        self.gridLayout_68.addLayout(self.horizontalLayout_69, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_34, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_76 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_76.setObjectName("horizontalLayout_76")
        self.verticalLayout_97 = QtWidgets.QVBoxLayout()
        self.verticalLayout_97.setObjectName("verticalLayout_97")
        self.groupBox_43 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_43.setObjectName("groupBox_43")
        self.verticalLayout_98 = QtWidgets.QVBoxLayout(self.groupBox_43)
        self.verticalLayout_98.setObjectName("verticalLayout_98")
        self.radio_classcorr_6 = QtWidgets.QRadioButton(self.groupBox_43)
        self.radio_classcorr_6.setChecked(False)
        self.radio_classcorr_6.setObjectName("radio_classcorr_6")
        self.buttonGroup.addButton(self.radio_classcorr_6)
        self.verticalLayout_98.addWidget(self.radio_classcorr_6)
        self.horizontalLayout_77 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_77.setObjectName("horizontalLayout_77")
        self.label_74 = QtWidgets.QLabel(self.groupBox_43)
        self.label_74.setEnabled(False)
        self.label_74.setAlignment(QtCore.Qt.AlignCenter)
        self.label_74.setObjectName("label_74")
        self.horizontalLayout_77.addWidget(self.label_74)
        self.comboBox_31 = QtWidgets.QComboBox(self.groupBox_43)
        self.comboBox_31.setEnabled(False)
        self.comboBox_31.setObjectName("comboBox_31")
        self.horizontalLayout_77.addWidget(self.comboBox_31)
        self.verticalLayout_98.addLayout(self.horizontalLayout_77)
        self.verticalLayout_97.addWidget(self.groupBox_43)
        spacerItem4 = QtWidgets.QSpacerItem(20, 278, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_97.addItem(spacerItem4)
        self.horizontalLayout_76.addLayout(self.verticalLayout_97)
        self.widget_corr_6 = QMatplotlibWidget(self.tab)
        self.widget_corr_6.setObjectName("widget_corr_6")
        self.horizontalLayout_76.addWidget(self.widget_corr_6)
        self.gridLayout_4.addLayout(self.horizontalLayout_76, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_81 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_81.setObjectName("horizontalLayout_81")
        self.verticalLayout_103 = QtWidgets.QVBoxLayout()
        self.verticalLayout_103.setObjectName("verticalLayout_103")
        self.groupBox_45 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_45.setObjectName("groupBox_45")
        self.verticalLayout_104 = QtWidgets.QVBoxLayout(self.groupBox_45)
        self.verticalLayout_104.setObjectName("verticalLayout_104")
        self.radio_classcorr_8 = QtWidgets.QRadioButton(self.groupBox_45)
        self.radio_classcorr_8.setChecked(False)
        self.radio_classcorr_8.setObjectName("radio_classcorr_8")
        self.buttonGroup.addButton(self.radio_classcorr_8)
        self.verticalLayout_104.addWidget(self.radio_classcorr_8)
        self.horizontalLayout_83 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_83.setObjectName("horizontalLayout_83")
        self.label_77 = QtWidgets.QLabel(self.groupBox_45)
        self.label_77.setEnabled(False)
        self.label_77.setAlignment(QtCore.Qt.AlignCenter)
        self.label_77.setObjectName("label_77")
        self.horizontalLayout_83.addWidget(self.label_77)
        self.comboBox_33 = QtWidgets.QComboBox(self.groupBox_45)
        self.comboBox_33.setEnabled(False)
        self.comboBox_33.setObjectName("comboBox_33")
        self.horizontalLayout_83.addWidget(self.comboBox_33)
        self.verticalLayout_104.addLayout(self.horizontalLayout_83)
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_45)
        self.radioButton.setObjectName("radioButton")
        self.buttonGroup.addButton(self.radioButton)
        self.verticalLayout_104.addWidget(self.radioButton)
        self.horizontalLayout_82 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_82.setObjectName("horizontalLayout_82")
        self.verticalLayout_104.addLayout(self.horizontalLayout_82)
        self.verticalLayout_103.addWidget(self.groupBox_45)
        spacerItem5 = QtWidgets.QSpacerItem(20, 278, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_103.addItem(spacerItem5)
        self.horizontalLayout_81.addLayout(self.verticalLayout_103)
        self.widget_corr_7 = QMatplotlibWidget(self.tab_2)
        self.widget_corr_7.setObjectName("widget_corr_7")
        self.horizontalLayout_81.addWidget(self.widget_corr_7)
        self.gridLayout_5.addLayout(self.horizontalLayout_81, 0, 0, 1, 1)
        self.tabWidget_4.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget_4, 3, 0, 1, 3)
        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget_4.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Timeseries Analysis"))
        self.groupBox.setTitle(_translate("Form", "Visualizer"))
        self.label_71.setText(_translate("Form", "Metadata"))
        self.pushButton_4.setText(_translate("Form", "Browse"))
        self.label_70.setText(_translate("Form", "Spectra"))
        self.pushButton_5.setText(_translate("Form", "Browse"))
        self.label_72.setText(_translate("Form", "Output"))
        self.pushButton_6.setText(_translate("Form", "Save"))
        self.groupBox_34.setTitle(_translate("Form", "Category"))
        self.rb_sep_all_5.setText(_translate("Form", "All Spectra (Mean && Standard Deviation)"))
        self.rb_sep_compare_5.setText(_translate("Form", "Compare Two Class Spectra"))
        self.radioClassDistribute.setText(_translate("Form", "Continuum Removal"))
        self.label_73.setText(_translate("Form", "Baseline"))
        self.x1CoordEdit_11.setText(_translate("Form", "0.93"))
        self.groupBox_42.setTitle(_translate("Form", "Compare"))
        self.label_58.setText(_translate("Form", "Class A"))
        self.label_59.setText(_translate("Form", "Class B"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_29), _translate("Form", "Spectral View"))
        self.groupBox_35.setTitle(_translate("Form", "Category"))
        self.radioFeatureVar.setText(_translate("Form", "Band Plot"))
        self.label_75.setText(_translate("Form", "Wavelength"))
        self.x1CoordEdit_13.setText(_translate("Form", "400"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_44), _translate("Form", "Distribution"))
        self.groupBox_38.setTitle(_translate("Form", "Methods"))
        self.radioFTEST_5.setText(_translate("Form", "Spectral Discrimination Index"))
        self.radioSEP_5.setText(_translate("Form", "Oneway Anova"))
        self.radioMUTUAL_5.setText(_translate("Form", "Tukey’s multi-comparison"))
        self.label_67.setText(_translate("Form", "Wavelength"))
        self.x1CoordEdit_10.setText(_translate("Form", "400"))
        self.radioTREE_5.setText(_translate("Form", "Kruskal Wallis Non-Parametric H Test"))
        self.groupBox_39.setTitle(_translate("Form", "Compare"))
        self.label_63.setText(_translate("Form", "Class A"))
        self.label_64.setText(_translate("Form", "Class B"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_33), _translate("Form", "Band Separability"))
        self.groupBox_40.setTitle(_translate("Form", "Category"))
        self.radio_classcorr_5.setText(_translate("Form", "Band Correlation"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_34), _translate("Form", "Band Correlation"))
        self.groupBox_43.setTitle(_translate("Form", "Category"))
        self.radio_classcorr_6.setText(_translate("Form", "Statistics"))
        self.label_74.setText(_translate("Form", "Parameter"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab), _translate("Form", "Class Vs Property"))
        self.groupBox_45.setTitle(_translate("Form", "Category"))
        self.radio_classcorr_8.setText(_translate("Form", "Single parameter"))
        self.label_77.setText(_translate("Form", "Parameter"))
        self.radioButton.setText(_translate("Form", "All Parameter"))
        self.tabWidget_4.setTabText(self.tabWidget_4.indexOf(self.tab_2), _translate("Form", "Metadata_Statistics"))

from ProximalAnalysisTool.Ui.qmatplotlibwidget import QMatplotlibWidget