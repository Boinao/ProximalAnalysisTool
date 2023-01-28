# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:04:20 2019

@author: Trainee
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:51:58 2019

@author: ross
"""


import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

import logging
import datetime
import platform
import traceback
from time import strftime
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QThread, Qt
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QSplashScreen, QMainWindow , QVBoxLayout
from PyQt5.QtGui import QPixmap, QTextCursor

# import matplotlib
# matplotlib.use('Qt5Agg')
# from matplotlib import rcParams
# rcParams['legend.fontsize'] = 8
# from pylab import plot, xlabel, legend, ion, close, draw, array, clf,show
# ion()

try:
    from Ui.MainwindowUi import Ui_MainWindow
    from utils import delete

    from modules.IndicesASD import IndicesASD
    from modules.DimensionReduction import DimensionReduction
    from modules.ClassificationDistances import spectralDistance
    from modules.ClassificationSupervised import ClassificationSupervised
    from modules.Visualizer import Visualizer
    from modules.FGCC import Fgcc
    from modules.ViewDataASD import ViewDataASD
    from modules.ResamplingData import Resample
    from modules.ResamplingSpectral import Resampling
    from modules.Sig_file_viewer_OLD import Sigreader
    from modules.ViewDataSVC import ViewDataSVC
    from modules.ViewDataSEV import  ViewDataSEV
    from modules.Timeseries import Timeseries
    from modules.IndicesSVC import IndicesSVC
    from modules.SimulationPy6s import Py6s
    from modules.IndicesGeneric import IndicesGeneric
    from modules.SimulationProsail import Prosail_Simulation
    # from RegressionTrain import RegressionTrain
    from specdal.containers.spectrum import Spectrum
    from specdal.containers.collection import Collection
    from modules.RegressionUnivariate import UnivariateRegression
    from modules.RegressionMultivariate import Multi_Regression
    from modules.TransformPreProcessing import Preprocess_Transoform
    from modules.SimulationMixing import SimulationMixing
    from modules.Spectra_library_search import SpectraLibrarySearch
    from modules.Spectra_library_match import  SpectraLibraryMatch
    from modules.excepthook import my_exception_hook
except Exception as e:
    import traceback
    print(e, traceback.format_exc())
#class Stream(QtCore.QObject):
#    newText=QtCore.pyqtSignal(str)
#    def write(self,text):
#        self.newText.emit(str(text))


file_path = os.path.split(os.path.dirname(__file__))[0]


logger=logging.getLogger(__name__)

fmt='%(asctime)s | %(levelname)8s | %(message)s'

today=datetime.date.today()

logpath = os.path.join(os.path.join(os.environ['USERPROFILE']), 'SDAT\\logs')
if not os.path.exists(logpath):
    os.makedirs(logpath)
logfilename=os.path.join(logpath,'SDAT_app_{}.log'.format(today.strftime('%Y_%m_%d')))
file_handler=logging.FileHandler(logfilename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(file_handler)

def clearLayout(layout):
    if layout != None:
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clearLayout(child.layout())



def del_layout(qlayout, idx):
    """
    This method will delete the n - idxed layout that exists in qlayout

    +--------------------+
    | +----------------+ | The outer box is QLayout
    | |       0        | | The inner Layouts marked as 0 and 1 can be indexed by qlayout
    | |                | | You can reference the index of these inner layouts by using
    | +----------------+ | the method takeAt()
    | +----------------+ | In this case if I want to access Layout 2
    | |       1        | | I use qlayout.takeAt(1)
    | |                | |
    | +----------------+ |
    +--------------------+


    :param qlayout:
    :param idx:
    :return:
    """
    to_delete = qlayout.takeAt(0)  # remove the layout item at n-1 index
    if to_delete is not None:  # We run this method as long as there are objects
        while to_delete.count():  # while the count is not 0
            item = to_delete.takeAt(0)  # grab the layout item at 0th index
            widget = item.widget()  # get the widget at this location
            if widget is not None:  # if there is an object in this widget
                widget.deleteLater()  # delete this widget
            else:
                pass


def del_qwidget(QWidget):
    QWidget.close()
    QWidget.deleteLater()



class EmittingStream(QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

class MainWindow(Ui_MainWindow,QThread):

    taskFinished = QtCore.pyqtSignal()

    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)
#        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QSizePolicy.Fixed)
#        self.setupUi(self)
#        self.insert_idx=0
        self.debug=True
        self.widgetList = []




    def setupUi(self, MainWindowUi):
        super(MainWindow,self).setupUi(MainWindowUi)
        self.MainWindow = MainWindowUi
        self.scrollArea.setStyleSheet(stylesheet)
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 1562, 150))
#        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.connectWidgets()
        self.normal_mode()
        self.MainWindow.closeEvent = self.closeEvent

        # print("SDAT path is in: "+str(pluginPath))

    def closeEvent(self, event):
        """
        Write window size and position to config file, or system registry
        """
        # self._writeWindowAttributeSettings()
        event.accept()


    def connectWidgets(self):
        """
        Connect all the widgets associated with the MainWindow UI

        :return:
        """
        try:
            ### Data Management
            self.actionView_Data.triggered.connect(lambda: self.addWidget(ViewDataASD))
            self.actionNewSigreader.triggered.connect(lambda: self.addWidget(ViewDataSVC))
            self.actionspectraevolutionreader.triggered.connect(lambda: self.addWidget(ViewDataSEV))

            # Preprocessing
            self.actionLoad_Preprocessing.triggered.connect(lambda: self.addWidget(Preprocess_Transoform))
            self.actionNewResampling.triggered.connect(lambda: self.addWidget(Resampling))
            self.actionVisualizer.triggered.connect(lambda: self.addWidget(Visualizer,1))

            #Spectral Indices
            self.actionSpectral_Indices.triggered.connect(lambda: self.addWidget(IndicesASD))
            self.actionSVCSpectral_Indices.triggered.connect(lambda: self.addWidget(IndicesSVC))
            self.actionCSVSpectral_Indices.triggered.connect(lambda: self.addWidget(IndicesGeneric))

            # UnivariateRegression
            self.actionUnivariate_Regression.triggered.connect(lambda: self.addWidget(UnivariateRegression))
            self.actionMultivariate_Regression.triggered.connect(lambda: self.addWidget(Multi_Regression))
            
            # Dimesnion Reduction

            # self.actionFeature_Selection.triggered.connect(lambda: self.addWidget(FeatureSelection))
            # self.actionMultiple_Classes.triggered.connect(lambda: self.addWidget(FeatureSelectionMultipleClasses))
            self.actionDimension_Reduction.triggered.connect(lambda: self.addWidget(DimensionReduction))


            #ClassificationSupervised
            self.actionSpectral_Distance.triggered.connect(lambda: self.addWidget(spectralDistance))
            self.actionClassification.triggered.connect(lambda: self.addWidget(ClassificationSupervised))

            #Time Series
            self.actionTimeSeries.triggered.connect(lambda: self.addWidget(Visualizer,2))

            #Image Utility
            self.actionFgcc.triggered.connect(lambda: self.addWidget(Fgcc))

            #Simulation
            self.actionSpectraSimulation.triggered.connect(lambda: self.addWidget(SimulationMixing))
            self.actionPy6sTimeSeries.triggered.connect(lambda: self.addWidget(Py6s))
            self.actionProsail.triggered.connect(lambda: self.addWidget(Prosail_Simulation))

            #Spectral Library
            self.actionSpectralLibrarySearch.triggered.connect(lambda: self.addWidget(SpectraLibrarySearch))
            self.actionSpectralLibraryMatch.triggered.connect(lambda: self.addWidget(SpectraLibraryMatch))



            self.okPushButton.clicked.connect(self.on_okButton_clicked)
            self.deletePushButton.clicked.connect(self.on_delete_module_clicked)
            self.stopPushButton.clicked.connect(self.on_stopButton_clicked)
            self.actionMenuAbout.triggered.connect(self.showAbout)

#
        except Exception as e:
            print(e)

    def showAbout(self):

        from modules.AboutSDAT import AboutSDAT
        d = AboutSDAT()
        d.exec_()
        del d


    def normal_mode(self):
        """
        Change the direction of stdout to print to the UI console instead
        :return:
        """
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = sys.__stderr__

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def checkForBg(self):

        if len(self.widgetList) > 0:
            self.scrollArea.setStyleSheet("")
        else:
            self.scrollArea.setStyleSheet(stylesheet)

    def addWidget(self, obj, kind=None):
        """
        Organize our widgets using a list
        Each widget is addressed separately due to being in a list
        :param obj:
        :return:
        """
        self.scrollArea.setStyleSheet("")
        if len(self.widgetList) > 0:
            self.on_delete_module_clicked()
        #print (self.widgetList)
        idx=-1

        Form = QWidget()
        self.widgetList.append(obj())

        if kind is not None:
            self.widgetList[idx].setupUi(Form,kind)
        else:
            self.widgetList[idx].setupUi(Form)
        # self.widgetList[idx].setupUi(self.centralwidget)
        self.widgetLayout = QVBoxLayout()
        self.widgetLayout.setObjectName("widgetLayoutAgain")
        self.verticalLayout_2.insertLayout(idx, self.widgetLayout)
        self.widgetLayout.addWidget(self.widgetList[idx].get_widget())

        self.checkForBg()



        # self.verticalLayout_2.addWidget(self.widgetList[idx].get_widget())
        # this should scroll the view all the way down after adding the new widget.
        scrollbar = self.scrollArea.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_delete_module_clicked(self):
        """
        Check to see if the last item is enabled
        If it is, delete the last item in the list
        And then call the del_layout function, which will remove the item from the UI
        :return:
        """
        self.textBrowser.clear()
        try:
            if len(self.widgetList)>0:
                if self.widgetList[-1].isEnabled():
                    # self.widgetList[-1].delete()
                    del self.widgetList[-1]
                    delete.del_layout(self.verticalLayout_2)
                    self.scrollArea.setStyleSheet(stylesheet)

                else:
                    print("Cannot delete")
            else:
                QtWidgets.QMessageBox.information(None, 'Information', "No modules to reset!", QtWidgets.QMessageBox.Ok)

        except Exception as e:
            import traceback
            print(e, traceback.format_exc())

#    def clear(self):
#        """
#        Delete all modules in GUI
#        :return:
#        """
#        while len(self.widgetList) > 0 and self.widgetList[-1].isEnabled():
#            self.on_delete_module_clicked()
#        self.title.setFileName('')
#        self.MainWindow.setWindowTitle(self.title.display())
#        self.textBrowser.clear()


    def runModules(self):

        """
        This function iterates through a list of object addresses
        which then run its dot notated run()

        iterate through our widgets, start from the last left off item
        get the name of our current widget item
        start the timer
        print the name of the module running
        run our current module's function()
        get our end time
        print how long it took our current module to execute based on start time and end time
        disable our current module
        increment our left off module

        :return:
        """
        self.textBrowser.clear()
        for modules in range(len(self.widgetList)):
            name_ = type(self.widgetList[modules]).__name__
            s = time.time()
            print("{} Module is Running...".format(name_))
#            self.textBrowser.append(format(name_)+" Module is Running...")
            self.widgetList[modules].run()
            e = time.time()
            print("Module {} executed in: {} seconds".format(name_, round(e - s,4)))
#            self.textBrowser.append("\n"+format(name_)+"Module executed in "+format(e - s)+" seconds")

#            self.widgetList[modules].setDisabled(True)
#            DisplayData=self.widgetList[modules].printOutput()
#            self.textBrowser.append(DisplayData)



    def _logger(self, function):
        try:
            function()
        except Exception as e:
            # print("Your {} module broke with error: {}.".format(type(self.widgetList[0]).__name__, e))
            print("Your {} module broke with error: {}.".format(type(self.widgetList[0]).__name__,
                                                                traceback.format_exc()))
#            self.widgetList[0].setDisabled(False)

    def _exceptionLogger(self, function):
        """
        Logs an exception that occurs during the running of a function
        Take a function address in as an input

        :param function:
        :return:
        """
        logfile = "file"


        try:
            function()
        except Exception as e:

            logger.error("Your {} module broke with error: {}.".format( type(self.widgetList[0]).__name__, traceback.format_exc()))
            # logging.exception('[%s %s] (%s) :' % (timenow,type(self.widgetList[0]).__name__, traceback.format_exc()))
            print("Your {} module broke with error: {}.".format(type(self.widgetList[0]).__name__, e))
            print('\nException was logged to "%s"' % (os.path.join(logpath, logfilename)))


    def on_okButton_clicked(self):
        """
        Start the multithreading function

        :return:
        """

#        self.onStart()
#        self.taskFinished.connect(self.onFinished)
#        print("Hi after Run Button Clicked ")
#        QApplication.processEvents()
        self.progressBar.setRange(0, 0)
        try:
            self.run()
        except:
            pass

        self.progressBar.setRange(0, 1)
        self.progressBar.setValue(1)




    def on_stopButton_clicked(self):
        """
        Hard terminate running thread

        Technically you should never do this.
        But given that some tasks are monumentally huge,
        I feel that it is justified.

        :return:
        """

        if self.isRunning():
            self.terminate()
            self.taskFinished.emit()
        else:
            print("There is nothing running right now")

    def run(self):
        """
        Start the thread for running all the modules

        :return:
        """
#        print("just hi")
        if self.debug:
            self._exceptionLogger(self.runModules)
            # self._logger(self.runModules)
        else:
            self._logger(self.runModules)
        self.taskFinished.emit()

    def onStart(self):
        """
        This is multithreading thus run() == start()
        make the bar pulse green
        TaskThread.start()

        :return:
        """

        self.progressBar.setRange(0, 0)
        self.TaskThread()

    def onFinished(self):
        """
        When a given task is finished
        stop the bar pulsing green
        and display 100% for bar.

        :return:
        """
        self.progressBar.setRange(0, 1)
        self.progressBar.setValue(1)


    def on_insertModule(self):
        self.insert_idx += 1



def get_splash(app):
    """
    Get the splash screen for the application
    But check to see if the image even exists

    :param app:
    :return:
    """
    try:
        dirs = ['.','./images/','../images/','../../images/','./SDAT/images/','../SDAT/images/','ProximalAnalysisTool/images/']
        pluginPath = os.path.split(os.path.dirname(__file__))[0]
        # print(pluginPath, flush=True)
        for dir in dirs:
            # if os.path.exists(dir + 'SDAT.JPG'):
            #     splash_pix = QPixmap(dir + 'SDAT.JPG')  # default
            if os.path.exists(os.path.join(pluginPath, dir, 'SDAT.JPG')):
                splash_pix = QPixmap(os.path.join(pluginPath, dir, 'SDAT.JPG'))  # default
                splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
                splash.setMask(splash_pix.mask())
                splash.show()
                time.sleep(2.5)
                app.processEvents()
                break
    except Exception as e:
        pass
    return 0


stylesheet="""
        QWidget {
            background-image:url('./images/sdat_banner.PNG');
            background-repeat:no-repeat;
            background-position: center;
            }"""

# import pyi_splash
def main():
    # sys._excepthook = sys.excepthook
    # sys.excepthook = my_exception_hook
    app = QApplication(sys.argv)
    get_splash(app)
    # pyi_splash.update_text("Loading SDAT modules...")
    # time.sleep(1.5)
    # pyi_splash.close()
    mainWindow = QMainWindow()
    ui = MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

