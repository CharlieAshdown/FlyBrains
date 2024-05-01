from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QDir
import sys
import os

from video_visualiser import video_visualiser


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('UI/Video_Visualiser_Main.ui', self) # Load the .ui file
        self.filepaths = []
        self.dirpath = QDir.currentPath()
        self.dragDrop = self.findChild(QtWidgets.QWidget, 'dragdropwidget')

        self.choose_rgb = self.findChild(QtWidgets.QComboBox, 'comboBox')

        self.array_len_label = self.findChild(QtWidgets.QLabel, 'label')
        self.lct_label = self.findChild(QtWidgets.QLabel, 'label_2')
        self.file_label = self.findChild(QtWidgets.QLabel, 'label_3')

        self.dragDrop.setAcceptDrops(True)
        self.file_button = self.dragDrop.findChild(QtWidgets.QPushButton, 'pushButton_2')
        self.file_button.clicked.connect(self.getFile)

        self.sim_button = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.sim_button.clicked.connect(self.run_sim)

        self.save_images_box = self.findChild(QtWidgets.QCheckBox, 'checkBox')
        self.brighten_box = self.findChild(QtWidgets.QCheckBox, 'checkBox_2')
        self.one_channel_box = self.findChild(QtWidgets.QCheckBox, 'checkBox_3')
        self.one_channel_box.clicked.connect(self.enable_choose_rgb)

        self.show()  # Show the GUI

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.filepaths += [u.toLocalFile() for u in event.mimeData().urls()]
        self.file_label.setText(f"File Names = {str([os.path.split(u)[-1] for u in self.filepaths])}")

    def getFile(self):
        self.filepaths.append(QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                                   directory=self.dirpath))
        self.file_label.setText(f"File Names = {str([os.path.split(u)[-1] for u in self.filepaths])}")

    def enable_choose_rgb(self):
        if self.one_channel_box.isChecked():
            self.choose_rgb.setEnabled(True)
        else:
            self.choose_rgb.setEnabled(False)

    def run_sim(self):
        if len(self.filepaths) == 0:
            return
        for file in self.filepaths:
            video_visualiser(path=file,
                             brighten=self.brighten_box.isChecked(),
                             black_and_white=self.one_channel_box.isChecked(),
                             save_images=self.save_images_box.isChecked(),
                             channel=self.choose_rgb.currentText().lower())
        self.filepaths = []
        self.file_label.setText(f"File Names = {str([os.path.split(u)[-1] for u in self.filepaths])}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec()


