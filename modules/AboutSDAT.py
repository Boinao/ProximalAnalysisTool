# -*- coding: utf-8 -*-

"""
***************************************************************************
    AboutSDAT.py
    ---------------------
    Date                 : November 2020
    Author               : Anand SS, Ross, Nidhin
    Email                : anandss@isro.gov.in
***************************************************************************
"""

import sys
import os
from PyQt5.QtWidgets import QDialog
from Ui.About_SDATUi import Ui_Dialog
pluginPath = os.path.split(os.path.dirname(__file__))[0]
POSTFIX = '_AboutSDAT'
import subprocess

class AboutSDAT(QDialog, Ui_Dialog):
    '''
    Returns the technical documentation and other supportive document
    related to SDAT Software.
    '''
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("About SDAT")
        self.connectWidgets()

    def connectWidgets(self):
        self.pushButton_2.clicked.connect(lambda: self.browseButton_clicked())
        self.pushButton_3.clicked.connect(lambda: self.asd_doc_clicked())
        self.pushButton_4.clicked.connect(lambda: self.svc_doc_clicked())
        self.pushButton_5.clicked.connect(lambda: self.SE_doc_clicked())
        self.pushButton_6.clicked.connect(lambda: self.CalVal_doc_clicked())
        self.pushButton_7.clicked.connect(lambda: self.specFieldGuide())

    def specFieldGuide(self):
        filepath = os.path.join(pluginPath, 'external/about', 'SPEC_GUIDE.pdf')
        print(filepath)
        subprocess.Popen([filepath], shell=True)

    def browseButton_clicked(self):
        filepath = os.path.join(pluginPath, 'external/about', 'SDAT_TECHNICAL_DOC.pdf')
        subprocess.Popen([filepath], shell=True)

    def asd_doc_clicked(self):
        filepath = os.path.join(pluginPath, 'external/about', 'ASD_MANUAL.pdf')
        subprocess.Popen([filepath], shell=True)

    def svc_doc_clicked(self):
        filepath = os.path.join(pluginPath, 'external/about', 'SVC_MANUAL.pdf')
        subprocess.Popen([filepath], shell=True)

    def SE_doc_clicked(self):
        filepath = os.path.join(pluginPath, 'external/about', 'SEV_BRO.pdf')
        subprocess.Popen([filepath], shell=True)

    def CalVal_doc_clicked(self):
        filepath = os.path.join(pluginPath, 'external/about', 'CALVAL_HB.pdf')
        subprocess.Popen([filepath], shell=True)
