# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Proximal\ProximalAnalysisTool\Ui\ClassificationUi.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(701, 563)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_2.addWidget(self.lineEdit_3, 0, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.pushButton, 0, 2, 1, 1)
        self.metadataLbl = QtWidgets.QLabel(self.groupBox)
        self.metadataLbl.setObjectName("metadataLbl")
        self.gridLayout_2.addWidget(self.metadataLbl, 1, 0, 1, 1)
        self.metadataTxt = QtWidgets.QLineEdit(self.groupBox)
        self.metadataTxt.setObjectName("metadataTxt")
        self.gridLayout_2.addWidget(self.metadataTxt, 1, 1, 1, 1)
        self.browseMetaBtn = QtWidgets.QPushButton(self.groupBox)
        self.browseMetaBtn.setObjectName("browseMetaBtn")
        self.gridLayout_2.addWidget(self.browseMetaBtn, 1, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 2, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_2.addWidget(self.lineEdit_4, 2, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_2.addWidget(self.pushButton_2, 2, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 3, 0, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_2.addWidget(self.lineEdit_5, 3, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 4, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout_2.addWidget(self.comboBox, 4, 1, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_2.addWidget(self.checkBox, 5, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 6, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_2.addWidget(self.lineEdit_2, 6, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 7, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 7, 1, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.gridLayout.addWidget(self.comboBox_3, 0, 1, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.gridLayout.addWidget(self.comboBox_2, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_2, 8, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 9, 0, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Classification"))
        self.label_2.setText(_translate("Form", "Input:"))
        self.pushButton.setText(_translate("Form", "Browse"))
        self.metadataLbl.setText(_translate("Form", "Metdata"))
        self.browseMetaBtn.setText(_translate("Form", "Browse"))
        self.label_4.setText(_translate("Form", "Output:"))
        self.pushButton_2.setText(_translate("Form", "Save As"))
        self.label_7.setText(_translate("Form", "Test Size:"))
        self.lineEdit_5.setText(_translate("Form", "0.3"))
        self.label_6.setText(_translate("Form", "Method:        "))
        self.comboBox.setItemText(0, _translate("Form", "SVM RBF"))
        self.comboBox.setItemText(1, _translate("Form", "SVM Linear"))
        self.comboBox.setItemText(2, _translate("Form", "SVM Poly"))
        self.comboBox.setItemText(3, _translate("Form", "Random Forest"))
        self.comboBox.setItemText(4, _translate("Form", "Gausian Mixture Model"))
        self.comboBox.setItemText(5, _translate("Form", "KNN"))
        self.comboBox.setItemText(6, _translate("Form", "Multinomial Logistic"))
        self.checkBox.setText(_translate("Form", "PCA"))
        self.label.setText(_translate("Form", "Enter C Value:"))
        self.lineEdit_2.setText(_translate("Form", "1"))
        self.label_3.setText(_translate("Form", "Enter Gamma :"))
        self.lineEdit.setText(_translate("Form", "0.01"))
        self.groupBox_2.setTitle(_translate("Form", "Parameter Selection"))
        self.comboBox_3.setItemText(0, _translate("Form", "Multiclass Options"))
        self.comboBox_3.setItemText(1, _translate("Form", "ovr"))
        self.comboBox_3.setItemText(2, _translate("Form", "multinomial"))
        self.comboBox_2.setItemText(0, _translate("Form", "Solver Options"))
        self.comboBox_2.setItemText(1, _translate("Form", "newton-cg"))
        self.comboBox_2.setItemText(2, _translate("Form", "lbfgs"))
        self.comboBox_2.setItemText(3, _translate("Form", "sag"))

