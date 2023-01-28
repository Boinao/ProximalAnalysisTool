from PyQt5 import QtCore
import pandas as pd
# from PyQt5.QtCore import *
class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid() and (role == QtCore.Qt.DisplayRole):
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[col]
        return None
    
    def flags(self,index):
        f=super(self.__class__,self).flags(index)
        f|=QtCore.Qt.ItemIsEditable
        f|=QtCore.Qt.ItemIsSelectable
        f|=QtCore.Qt.ItemIsEnabled
        f|=QtCore.Qt.ItemIsDragEnabled
        f|=QtCore.Qt.ItemIsDropEnabled
        return f
    
    def setData(self,index,value,role=QtCore.Qt.EditRole):
        if index.isValid():
            row=index.row()
            col=index.column()
            self._data.iloc[row][col]=value #float(value)
#            print(self._data.iloc[row][col])
            
            self.dataChanged.emit(index,index,(QtCore.Qt.DisplayRole,))
            return True
        return False

    def removeAllDataFrameRows(self):
        position=0
        rows=self.rowCount()
        self.beginRemoveRows(QtCore.QModelIndex(), position,position+rows-1)
        for idx,line in self._data.iterrows():
            self._data.drop(idx,inplace=True)

        self._data.reset_index(inplace=True, drop=True)
        self.endRemoveRows()


        
