# -*- coding: utf-8 -*-

"""
***************************************************************************
    qmatplotlibwidget.py
    ---------------------
    Date                 : July 2014
    Copyright            : (C) 2014-2017 by Alexander Bruy
    Email                : alexander dot bruy at gmail dot com
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

__author__ = 'Alexander Bruy'
__date__ = 'July 2014'
__copyright__ = '(C) 2014-2017, Alexander Bruy'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

import os

import os

from PyQt5.QtGui import QPalette, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QAction

import matplotlib

mplVersion = matplotlib.__version__.split('.')

from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

pluginPath = os.path.split(os.path.dirname(__file__))[0]


class QMatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.parent = parent

        self.figure = Figure()

        FigureCanvasQTAgg.__init__(self, self.figure)
        FigureCanvasQTAgg.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

        self.figure.canvas.setFocusPolicy(Qt.ClickFocus)
        self.figure.canvas.setFocus()

        # rcParams['font.serif'] = 'Verdana, Arial, Liberation Serif'
        # rcParams['font.sans-serif'] = 'Tahoma, Arial, Liberation Sans'
        # rcParams['font.cursive'] = 'Courier New, Arial, Liberation Sans'
        # rcParams['font.fantasy'] = 'Comic Sans MS, Arial, Liberation Sans'
        # rcParams['font.monospace'] = 'Courier New, Liberation Mono'


class QMatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        # self.canvas = QMatplotlibCanvas()
        # self.figure = self.canvas.figure

        self.figure = Figure()

        # rcParams['font.serif'] = 'Verdana, Arial, Liberation Serif'
        # rcParams['font.sans-serif'] = 'Tahoma, Arial, Liberation Sans'
        # rcParams['font.cursive'] = 'Courier New, Arial, Liberation Sans'
        # rcParams['font.fantasy'] = 'Comic Sans MS, Arial, Liberation Sans'
        # rcParams['font.monospace'] = 'Courier New, Liberation Mono'
        # rcParams['font.size'] = 12
        # rcParams['legend.fontsize'] = 12
        # rcParams['xtick.labelsize'] = 12
        # rcParams['ytick.labelsize'] = 12
        # # rcParams['figure.dpi'] = 100.0
        # rcParams['axes.titlesize'] = 12
        # rcParams['axes.labelsize'] = 12
        # rcParams['legend.frameon'] = True

        self.canvas = FigureCanvasQTAgg(self.figure)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.canvas.setSizePolicy(sizePolicy)

        self.ax = self.figure.add_subplot(111)
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        self.toolBar = NavigationToolbar2QT(self.canvas, self)

        #        bgColor = self.palette().color(QPalette.Background).name()
        #        self.figure.set_facecolor(bgColor)

        self.layout = QVBoxLayout()
        # self.layout.setSpacing(2)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.layout.addWidget(self.toolBar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

        self._setupToolbar()

    def _setupToolbar(self):
        self.actionToggleGrid = QAction(self.tr('Toggle grid'), self.toolBar)
        #        self.actionToggleGrid.setIcon(
        #            QIcon(os.path.join(pluginPath, 'icons', 'toggleGrid.svg')))
        self.actionToggleGrid.setCheckable(True)

        self.actionToggleGrid.triggered.connect(self.toggleGrid)

        self.toolBar.insertAction(self.toolBar.actions()[7], self.actionToggleGrid)
        self.toolBar.insertSeparator(self.toolBar.actions()[8])

    def toggleGrid(self):
        self.ax.grid()
        self.canvas.draw()

    def alignLabels(self):
        self.figure.autofmt_xdate()

    def clear(self):
        self.ax.clear()
        self.canvas.draw()

    def setTitle(self, text):
        self.ax.set_title(text)

    def setXAxisCaption(self, text):
        self.ax.set_xlabel(text)

    def setYAxisCaption(self, text):
        self.ax.set_ylabel(text)
