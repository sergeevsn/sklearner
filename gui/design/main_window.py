# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class CheckableComboBox(QtWidgets.QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(QtWidgets.QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = QtWidgets.qApp.palette()
        palette.setBrush(QtGui.QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == QtCore.Qt.Checked:
                    item.setCheckState(QtCore.Qt.Unchecked)
                else:
                    item.setCheckState(QtCore.Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == QtCore.Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QtGui.QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, QtCore.Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QtGui.QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        item.setData(QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == QtCore.Checked:
                res.append(self.model().item(i).data())
        return res


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(352, 782)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.groupBox_RegressionParams = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_RegressionParams.setGeometry(QtCore.QRect(10, 350, 331, 161))
        self.groupBox_RegressionParams.setFlat(False)
        self.groupBox_RegressionParams.setCheckable(False)
        self.groupBox_RegressionParams.setObjectName("groupBox_RegressionParams")

        self.label = QtWidgets.QLabel(self.groupBox_RegressionParams)
        self.label.setGeometry(QtCore.QRect(10, 30, 101, 16))
        self.label.setObjectName("label")

        self.comboBox_Regressor = QtWidgets.QComboBox(self.groupBox_RegressionParams)
        self.comboBox_Regressor.setGeometry(QtCore.QRect(80, 30, 241, 22))
        self.comboBox_Regressor.setObjectName("comboBox_Regressor")

        self.label_2 = QtWidgets.QLabel(self.groupBox_RegressionParams)
        self.label_2.setGeometry(QtCore.QRect(10, 60, 61, 16))
        self.label_2.setObjectName("label_2")

        self.comboBox_RegressorParameter = QtWidgets.QComboBox(self.groupBox_RegressionParams)
        self.comboBox_RegressorParameter.setGeometry(QtCore.QRect(80, 60, 241, 22))
        self.comboBox_RegressorParameter.setObjectName("comboBox_RegressorParameter")

        self.label_3 = QtWidgets.QLabel(self.groupBox_RegressionParams)
        self.label_3.setGeometry(QtCore.QRect(20, 90, 47, 13))
        self.label_3.setObjectName("label_3")

        self.lineEdit_RegressorParamMin = QtWidgets.QLineEdit(self.groupBox_RegressionParams)
        self.lineEdit_RegressorParamMin.setGeometry(QtCore.QRect(80, 90, 41, 20))
        self.lineEdit_RegressorParamMin.setObjectName("lineEdit_RegressorParamMin")

        self.label_4 = QtWidgets.QLabel(self.groupBox_RegressionParams)
        self.label_4.setGeometry(QtCore.QRect(130, 90, 47, 13))
        self.label_4.setObjectName("label_4")

        self.lineEdit_RegressorParamMax = QtWidgets.QLineEdit(self.groupBox_RegressionParams)
        self.lineEdit_RegressorParamMax.setGeometry(QtCore.QRect(190, 90, 41, 20))
        self.lineEdit_RegressorParamMax.setObjectName("lineEdit_RegressorParamMax")
        self.lineEdit_RegressorParamStep = QtWidgets.QLineEdit(self.groupBox_RegressionParams)
        self.lineEdit_RegressorParamStep.setGeometry(QtCore.QRect(280, 90, 41, 20))
        self.lineEdit_RegressorParamStep.setObjectName("lineEdit_RegressorParamStep")

        self.label_5 = QtWidgets.QLabel(self.groupBox_RegressionParams)
        self.label_5.setGeometry(QtCore.QRect(240, 90, 47, 13))
        self.label_5.setObjectName("label_5")

        self.button_RegressionParamsCheck = QtWidgets.QPushButton(self.groupBox_RegressionParams)
        self.button_RegressionParamsCheck.setGeometry(QtCore.QRect(260, 120, 61, 23))
        self.button_RegressionParamsCheck.setObjectName("button_RegressionParamsCheck")

        self.groupBox_ScalingParams = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_ScalingParams.setGeometry(QtCore.QRect(10, 230, 331, 101))
        self.groupBox_ScalingParams.setObjectName("groupBox_ScalingParams")

        self.checkBox_ScaleFeatures = QtWidgets.QCheckBox(self.groupBox_ScalingParams)
        self.checkBox_ScaleFeatures.setGeometry(QtCore.QRect(20, 30, 121, 17))
        self.checkBox_ScaleFeatures.setObjectName("checkBox_ScaleFeatures")

        self.checkBox_ScaleTarget = QtWidgets.QCheckBox(self.groupBox_ScalingParams)
        self.checkBox_ScaleTarget.setGeometry(QtCore.QRect(190, 30, 121, 17))
        self.checkBox_ScaleTarget.setObjectName("checkBox_ScaleTarget")

        self.label_6 = QtWidgets.QLabel(self.groupBox_ScalingParams)
        self.label_6.setGeometry(QtCore.QRect(10, 60, 47, 13))
        self.label_6.setObjectName("label_6")

        self.comboBox_Scaler = QtWidgets.QComboBox(self.groupBox_ScalingParams)
        self.comboBox_Scaler.setGeometry(QtCore.QRect(50, 60, 271, 22))
        self.comboBox_Scaler.setObjectName("comboBox_Scaler")

        self.groupBox_DataTableParams = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_DataTableParams.setGeometry(QtCore.QRect(10, 10, 331, 211))
        self.groupBox_DataTableParams.setObjectName("groupBox_DataTableParams")

        self.label_7 = QtWidgets.QLabel(self.groupBox_DataTableParams)
        self.label_7.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label_7.setObjectName("label_7")

        self.lineEdit_DataFileName = QtWidgets.QLineEdit(self.groupBox_DataTableParams)
        self.lineEdit_DataFileName.setGeometry(QtCore.QRect(80, 20, 221, 20))
        self.lineEdit_DataFileName.setObjectName("lineEdit_DataFileName")

        self.button_ChooseDataFile = QtWidgets.QToolButton(self.groupBox_DataTableParams)
        self.button_ChooseDataFile.setGeometry(QtCore.QRect(300, 20, 25, 21))
        self.button_ChooseDataFile.setObjectName("button_ChooseDataFile")

        self.button_ScanDataFile = QtWidgets.QPushButton(self.groupBox_DataTableParams)
        self.button_ScanDataFile.setGeometry(QtCore.QRect(260, 50, 61, 23))
        self.button_ScanDataFile.setObjectName("button_ScanDataFile")

        self.label_8 = QtWidgets.QLabel(self.groupBox_DataTableParams)
        self.label_8.setGeometry(QtCore.QRect(10, 80, 61, 16))
        self.label_8.setObjectName("label_8")

        self.comboBox_WellNameColumn = QtWidgets.QComboBox(self.groupBox_DataTableParams)
        self.comboBox_WellNameColumn.setGeometry(QtCore.QRect(80, 80, 241, 22))
        self.comboBox_WellNameColumn.setObjectName("comboBox_WellNameColumn")

        self.comboBox_TargetColumn = QtWidgets.QComboBox(self.groupBox_DataTableParams)
        self.comboBox_TargetColumn.setGeometry(QtCore.QRect(80, 140, 241, 22))
        self.comboBox_TargetColumn.setObjectName("comboBox_TargetColumn")

        self.label_9 = QtWidgets.QLabel(self.groupBox_DataTableParams)
        self.label_9.setGeometry(QtCore.QRect(10, 140, 61, 16))
        self.label_9.setObjectName("label_9")

        self.comboBox_FeaturesColumn = CheckableComboBox(self.groupBox_DataTableParams)
        self.comboBox_FeaturesColumn.setGeometry(QtCore.QRect(80, 110, 241, 22))
        self.comboBox_FeaturesColumn.setObjectName("comboBox_FeaturesColumn")

        self.label_10 = QtWidgets.QLabel(self.groupBox_DataTableParams)
        self.label_10.setGeometry(QtCore.QRect(10, 110, 61, 16))
        self.label_10.setObjectName("label_10")

        self.button_CheckTableParams = QtWidgets.QPushButton(self.groupBox_DataTableParams)
        self.button_CheckTableParams.setGeometry(QtCore.QRect(260, 170, 61, 23))
        self.button_CheckTableParams.setObjectName("button_CheckTableParams")
        self.button_CheckTableParams.setEnabled(False)

        self.groupBox_EvalParams = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_EvalParams.setGeometry(QtCore.QRect(10, 520, 331, 151))
        self.groupBox_EvalParams.setObjectName("groupBox_EvalParams")

        self.label_11 = QtWidgets.QLabel(self.groupBox_EvalParams)
        self.label_11.setGeometry(QtCore.QRect(10, 20, 51, 16))
        self.label_11.setObjectName("label_11")

        self.comboBox_SplitMode = QtWidgets.QComboBox(self.groupBox_EvalParams)
        self.comboBox_SplitMode.setGeometry(QtCore.QRect(80, 20, 241, 22))
        self.comboBox_SplitMode.setObjectName("comboBox_SplitMode")

        self.comboBox_TestWell = QtWidgets.QComboBox(self.groupBox_EvalParams)
        self.comboBox_TestWell.setGeometry(QtCore.QRect(80, 50, 241, 22))
        self.comboBox_TestWell.setObjectName("comboBox_TestWell")

        self.label_12 = QtWidgets.QLabel(self.groupBox_EvalParams)
        self.label_12.setGeometry(QtCore.QRect(10, 50, 51, 16))
        self.label_12.setObjectName("label_12")

        self.comboBox_EvalMode = QtWidgets.QComboBox(self.groupBox_EvalParams)
        self.comboBox_EvalMode.setGeometry(QtCore.QRect(80, 80, 241, 22))
        self.comboBox_EvalMode.setObjectName("comboBox_EvalMode")

        self.label_13 = QtWidgets.QLabel(self.groupBox_EvalParams)
        self.label_13.setGeometry(QtCore.QRect(10, 80, 51, 16))
        self.label_13.setObjectName("label_13")

        self.button_ChooseResultFolder = QtWidgets.QToolButton(self.groupBox_EvalParams)
        self.button_ChooseResultFolder.setGeometry(QtCore.QRect(300, 110, 25, 21))
        self.button_ChooseResultFolder.setObjectName("button_ChooseResultFolder")

        self.label_14 = QtWidgets.QLabel(self.groupBox_EvalParams)
        self.label_14.setGeometry(QtCore.QRect(10, 110, 71, 16))
        self.label_14.setObjectName("label_14")

        self.lineEdit_ResultsFolder = QtWidgets.QLineEdit(self.groupBox_EvalParams)
        self.lineEdit_ResultsFolder.setGeometry(QtCore.QRect(90, 110, 211, 20))
        self.lineEdit_ResultsFolder.setObjectName("lineEdit_ResultsFolder")

        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 710, 331, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")

        self.button_StartStop = QtWidgets.QPushButton(self.centralwidget)
        self.button_StartStop.setGeometry(QtCore.QRect(140, 680, 75, 23))
        self.button_StartStop.setObjectName("button_StartStop")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 352, 21))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_RegressionParams.setTitle(_translate("MainWindow", "Regression Parameters"))
        self.label.setText(_translate("MainWindow", "Regressor:"))
        self.label_2.setText(_translate("MainWindow", "Parameter:"))
        self.label_3.setText(_translate("MainWindow", "Minimum:"))
        self.label_4.setText(_translate("MainWindow", "Maximum:"))
        self.label_5.setText(_translate("MainWindow", "Step:"))
        self.button_RegressionParamsCheck.setText(_translate("MainWindow", "Check"))
        self.groupBox_ScalingParams.setTitle(_translate("MainWindow", "Scaling Parameters"))
        self.checkBox_ScaleFeatures.setText(_translate("MainWindow", "Scale Features"))
        self.checkBox_ScaleTarget.setText(_translate("MainWindow", "Scale Target"))
        self.label_6.setText(_translate("MainWindow", "Scaler:"))
        self.groupBox_DataTableParams.setTitle(_translate("MainWindow", "Data Table Parameters"))
        self.label_7.setText(_translate("MainWindow", "File Name:"))
        self.button_ChooseDataFile.setText(_translate("MainWindow", "..."))
        self.button_ScanDataFile.setText(_translate("MainWindow", "Scan"))
        self.label_8.setText(_translate("MainWindow", "Well Name:"))
        self.label_9.setText(_translate("MainWindow", "Target:"))
        self.label_10.setText(_translate("MainWindow", "Features:"))
        self.button_CheckTableParams.setText(_translate("MainWindow", "Check"))
        self.groupBox_EvalParams.setTitle(_translate("MainWindow", "Evaluation Parameters"))
        self.label_11.setText(_translate("MainWindow", "Split Mode:"))
        self.label_12.setText(_translate("MainWindow", "Test Well:"))
        self.label_13.setText(_translate("MainWindow", "Eval Mode:"))
        self.button_ChooseResultFolder.setText(_translate("MainWindow", "..."))
        self.label_14.setText(_translate("MainWindow", "Results Folder:"))
        self.button_StartStop.setText(_translate("MainWindow", "START"))
