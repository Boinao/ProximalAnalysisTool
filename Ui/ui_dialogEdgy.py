# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialogEdgy.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_HxToolsDialog(object):
    def setupUi(self, HxToolsDialog):
        HxToolsDialog.setObjectName(_fromUtf8("HxToolsDialog"))
        HxToolsDialog.resize(420, 476)
        self.verticalLayout = QtGui.QVBoxLayout(HxToolsDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(HxToolsDialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.inSelector = GdalToolsInOutSelector(HxToolsDialog)
        self.inSelector.setObjectName(_fromUtf8("inSelector"))
        self.gridLayout.addWidget(self.inSelector, 0, 2, 1, 1)
        self.label_3 = QtGui.QLabel(HxToolsDialog)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 2)
        self.outSelector = GdalToolsInOutSelector(HxToolsDialog)
        self.outSelector.setObjectName(_fromUtf8("outSelector"))
        self.gridLayout.addWidget(self.outSelector, 1, 2, 1, 1)
        self.label_4 = QtGui.QLabel(HxToolsDialog)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.SAM = QtGui.QRadioButton(HxToolsDialog)
        self.SAM.setChecked(True)
        self.SAM.setObjectName(_fromUtf8("SAM"))
        self.gridLayout.addWidget(self.SAM, 2, 1, 1, 2)
        self.ID = QtGui.QRadioButton(HxToolsDialog)
        self.ID.setObjectName(_fromUtf8("ID"))
        self.gridLayout.addWidget(self.ID, 3, 1, 1, 2)
        self.ED = QtGui.QRadioButton(HxToolsDialog)
        self.ED.setObjectName(_fromUtf8("ED"))
        self.gridLayout.addWidget(self.ED, 4, 1, 1, 2)
        self.BC = QtGui.QRadioButton(HxToolsDialog)
        self.BC.setObjectName(_fromUtf8("BC"))
        self.gridLayout.addWidget(self.BC, 5, 1, 1, 2)
        self.SID = QtGui.QRadioButton(HxToolsDialog)
        self.SID.setObjectName(_fromUtf8("SID"))
        self.gridLayout.addWidget(self.SID, 6, 1, 1, 2)
        self.verticalLayout.addLayout(self.gridLayout)
        self.checkBox_3 = QtGui.QCheckBox(HxToolsDialog)
        self.checkBox_3.setChecked(True)
        self.checkBox_3.setObjectName(_fromUtf8("checkBox_3"))
        self.verticalLayout.addWidget(self.checkBox_3)
        self.textBrowser = QtGui.QPlainTextEdit(HxToolsDialog)
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))
        self.verticalLayout.addWidget(self.textBrowser)
        self.progressBar = QtGui.QProgressBar(HxToolsDialog)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout.addWidget(self.progressBar)
        self.buttonBox = QtGui.QDialogButtonBox(HxToolsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)
        self.label.setBuddy(self.inSelector)

        self.retranslateUi(HxToolsDialog)
        QtCore.QMetaObject.connectSlotsByName(HxToolsDialog)

    def retranslateUi(self, HxToolsDialog):
        HxToolsDialog.setWindowTitle(_translate("HxToolsDialog", "Hyperspectral Edge Detection", None))
        self.label.setText(_translate("HxToolsDialog", "&Input file", None))
        self.label_3.setText(_translate("HxToolsDialog", "Output file", None))
        self.label_4.setText(_translate("HxToolsDialog", "Method", None))
        self.SAM.setText(_translate("HxToolsDialog", "Spectral Angle", None))
        self.ID.setText(_translate("HxToolsDialog", "Intensity Difference", None))
        self.ED.setText(_translate("HxToolsDialog", "Euclidean Distance", None))
        self.BC.setText(_translate("HxToolsDialog", "Bray Curtis Distance", None))
        self.SID.setText(_translate("HxToolsDialog", "Spectral Information Divergence", None))
        self.checkBox_3.setText(_translate("HxToolsDialog", "Load into Canvas when finished", None))

from .inOutSelector import GdalToolsInOutSelector
