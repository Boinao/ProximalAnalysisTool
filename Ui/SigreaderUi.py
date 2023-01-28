# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\Ui\SigreaderUi.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(677, 482)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.listWidget = QtWidgets.QListWidget(self.groupBox)
        self.listWidget.setObjectName("listWidget")
        self.gridLayout_3.addWidget(self.listWidget, 2, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 2)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_3.addWidget(self.pushButton_2, 1, 3, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_3.addWidget(self.pushButton, 0, 3, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_3.addWidget(self.lineEdit_2, 0, 2, 1, 1)
        self.tableView = QtWidgets.QTableView(self.groupBox)
        self.tableView.setObjectName("tableView")
        self.gridLayout_3.addWidget(self.tableView, 0, 4, 4, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_3.addWidget(self.lineEdit, 1, 2, 1, 1)
        self.svcSaveBtn = QtWidgets.QPushButton(self.groupBox)
        self.svcSaveBtn.setObjectName("svcSaveBtn")
        self.gridLayout_3.addWidget(self.svcSaveBtn, 5, 4, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout.setObjectName("gridLayout")
        self.svcRadRb = QtWidgets.QRadioButton(self.groupBox_3)
        self.svcRadRb.setObjectName("svcRadRb")
        self.gridLayout.addWidget(self.svcRadRb, 1, 0, 1, 1)
        self.svcReflRb = QtWidgets.QRadioButton(self.groupBox_3)
        self.svcReflRb.setChecked(True)
        self.svcReflRb.setObjectName("svcReflRb")
        self.gridLayout.addWidget(self.svcReflRb, 0, 0, 1, 1)
        self.svcWhiteRb = QtWidgets.QRadioButton(self.groupBox_3)
        self.svcWhiteRb.setObjectName("svcWhiteRb")
        self.gridLayout.addWidget(self.svcWhiteRb, 2, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_3, 0, 0, 1, 2)
        self.gridLayout_3.addWidget(self.groupBox_2, 3, 1, 3, 2)
        self.svcPlotBtn = QtWidgets.QPushButton(self.groupBox)
        self.svcPlotBtn.setObjectName("svcPlotBtn")
        self.gridLayout_3.addWidget(self.svcPlotBtn, 4, 4, 1, 1)
        self.groupBox_2.raise_()
        self.pushButton_2.raise_()
        self.tableView.raise_()
        self.label.raise_()
        self.pushButton.raise_()
        self.label_2.raise_()
        self.listWidget.raise_()
        self.lineEdit.raise_()
        self.lineEdit_2.raise_()
        self.svcSaveBtn.raise_()
        self.svcPlotBtn.raise_()
        self.horizontalLayout.addWidget(self.groupBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "SVC Spectroradiometer"))
        self.label.setText(_translate("Form", "Input"))
        self.label_2.setText(_translate("Form", "Output"))
        self.pushButton_2.setText(_translate("Form", "Browse"))
        self.pushButton.setText(_translate("Form", "Browse"))
        self.svcSaveBtn.setText(_translate("Form", "Save "))
        self.groupBox_2.setTitle(_translate("Form", "Options:"))
        self.groupBox_3.setTitle(_translate("Form", "Select:"))
        self.svcRadRb.setText(_translate("Form", "Radiance"))
        self.svcReflRb.setText(_translate("Form", "Reflectance"))
        self.svcWhiteRb.setText(_translate("Form", "White Reference"))
        self.svcPlotBtn.setText(_translate("Form", "Plot"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

