# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'inOutSelector.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GdalToolsInOutSelector(object):
    def setupUi(self, GdalToolsInOutSelector):
        GdalToolsInOutSelector.setObjectName("GdalToolsInOutSelector")
        GdalToolsInOutSelector.resize(294, 28)
        self.horizontalLayout = QtWidgets.QHBoxLayout(GdalToolsInOutSelector)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.fileEdit = QtWidgets.QLineEdit(GdalToolsInOutSelector)
        self.fileEdit.setMinimumSize(QtCore.QSize(100, 0))
        self.fileEdit.setObjectName("fileEdit")
        self.horizontalLayout.addWidget(self.fileEdit)
        self.combo = QtWidgets.QComboBox(GdalToolsInOutSelector)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.combo.sizePolicy().hasHeightForWidth())
        self.combo.setSizePolicy(sizePolicy)
        self.combo.setEditable(True)
        self.combo.setFrame(True)
        self.combo.setObjectName("combo")
        self.horizontalLayout.addWidget(self.combo)
        self.selectBtn = QtWidgets.QPushButton(GdalToolsInOutSelector)
        self.selectBtn.setObjectName("selectBtn")
        self.horizontalLayout.addWidget(self.selectBtn)

        self.retranslateUi(GdalToolsInOutSelector)
        QtCore.QMetaObject.connectSlotsByName(GdalToolsInOutSelector)

    def retranslateUi(self, GdalToolsInOutSelector):
        _translate = QtCore.QCoreApplication.translate
        self.selectBtn.setText(_translate("GdalToolsInOutSelector", "Select..."))

