import sys
import typing

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget

import design.main_window as des
sys.path.append("..")
from classlib import *

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == QtCore.Qt.Vertical:
                return str(self._data.index[section])

class TableWindow(QtWidgets.QWidget):
    def __init__(self, dataframe):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel("Table for learning")

        self.button = QtWidgets.QPushButton("OK")        
        self.button.clicked.connect(lambda:self.close())

        self.table = QtWidgets.QTableView()
        self.model = TableModel(dataframe)
        self.table.setModel(self.model)
        
        layout.addWidget(self.label)
        layout.addWidget(self.table)
        layout.addWidget(self.button)        
        self.setLayout(layout)


class SklearnerApp(QtWidgets.QMainWindow, des.Ui_MainWindow):

    def __init__(self):       
        super().__init__()
        self.setupUi(self)  
        self.learner = SklearnerLearn()
     
        self.button_ChooseDataFile.clicked.connect(self.chooseDataFile)
        self.button_ScanDataFile.clicked.connect(self.scanDataFile)
        
        self.button_CheckTableParams.clicked.connect(self.checkLearnTable)

        self.comboBox_FeaturesColumn.currentTextChanged.connect(self.updateFeaturesColumn)
        self.comboBox_WellNameColumn.currentTextChanged.connect(self.updateWellColumn)
        self.comboBox_TargetColumn.currentTextChanged.connect(self.updateTargetColumn)

    def errorMessage(self, text):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(text)       
        msg.setWindowTitle("Error")
        msg.exec_()       

    def checkLearnTable(self):        
        df_check = self.learner.table[[self.learner.well_column]+self.learner.feature_columns+[self.learner.target_column]]
        w = TableWindow(df_check)   
        w.setFixedSize(500,500) 
        w.show()

    def updateFeaturesColumn(self):
        self.learner.feature_columns = [s.strip() for s in self.comboBox_FeaturesColumn.currentText().split(',')]   

    def updateWellColumn(self):
        self.learner.well_column = self.comboBox_WellNameColumn.currentText()

    def updateTargetColumn(self):
        self.learner.target_column = self.comboBox_TargetColumn.currentText()       

    def chooseDataFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose data file", "", "CSV files (*.csv) ;; Excel files (*.xlsx)")
        if filename:
            self.lineEdit_DataFileName.setText(filename)   

    def scanDataFile(self):
        if not self.learner.read_learning_table(self.lineEdit_DataFileName.text()):
            self.errorMessage('Check data table file!')
            return
        self.comboBox_WellNameColumn.addItems(self.learner.table.columns)
        self.comboBox_FeaturesColumn.addItems(self.learner.table.columns)
        self.comboBox_TargetColumn.addItems(self.learner.table.columns)
        self.button_CheckTableParams.setEnabled(True)

        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = SklearnerApp()
    mainWindow.show()
    sys.exit(app.exec_())