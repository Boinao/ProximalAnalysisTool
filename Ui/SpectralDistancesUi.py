# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Proximal\ProximalAnalysisTool\Ui\SpectralDistancesUi.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(703, 595)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_2.addWidget(self.lineEdit_3)
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.metadatLbl = QtWidgets.QLabel(self.groupBox)
        self.metadatLbl.setObjectName("metadatLbl")
        self.verticalLayout.addWidget(self.metadatLbl)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.medataTxt = QtWidgets.QLineEdit(self.groupBox)
        self.medataTxt.setObjectName("medataTxt")
        self.horizontalLayout_3.addWidget(self.medataTxt)
        self.browseMetaBtn = QtWidgets.QPushButton(self.groupBox)
        self.browseMetaBtn.setObjectName("browseMetaBtn")
        self.horizontalLayout_3.addWidget(self.browseMetaBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_4.addWidget(self.lineEdit_4)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_4.addWidget(self.pushButton_2)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout.addWidget(self.comboBox)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        spacerItem = QtWidgets.QSpacerItem(20, 327, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.pushButton_2.raise_()
        self.comboBox.raise_()
        self.pushButton.raise_()
        self.label_5.raise_()
        self.lineEdit_4.raise_()
        self.lineEdit_3.raise_()
        self.medataTxt.raise_()
        self.browseMetaBtn.raise_()
        self.label_2.raise_()
        self.metadatLbl.raise_()
        self.label_4.raise_()
        self.label_3.raise_()
        self.horizontalLayout.addWidget(self.groupBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Spectral Distances"))
        self.label_2.setText(_translate("Form", "Input:"))
        self.pushButton.setText(_translate("Form", "Browse"))
        self.metadatLbl.setText(_translate("Form", "Metdata"))
        self.browseMetaBtn.setText(_translate("Form", "Browse"))
        self.label_4.setText(_translate("Form", "Output:"))
        self.pushButton_2.setText(_translate("Form", "Save As"))
        self.label_3.setText(_translate("Form", " Method:"))
        self.comboBox.setItemText(0, _translate("Form", "Select Method"))
        self.comboBox.setItemText(1, _translate("Form", "JM Distance"))
        self.comboBox.setItemText(2, _translate("Form", "Euclidean"))
        self.comboBox.setItemText(3, _translate("Form", "Manhattan"))
        self.comboBox.setItemText(4, _translate("Form", "Cosine"))
        self.comboBox.setItemText(5, _translate("Form", "Correlation"))
        self.comboBox.setItemText(6, _translate("Form", "Fractional"))
        self.comboBox.setItemText(7, _translate("Form", "Chebyshev"))
        self.comboBox.setItemText(8, _translate("Form", "SID"))
