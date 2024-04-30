from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QDir
import sys
import os
import glob
import torch
from time import time as t

from larvae_tracker import LarvaeTracker


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('UI/FlyBrains_Main.ui', self) # Load the .ui file
        self.filepaths = []
        self.dirpath = QDir.currentPath()
        self.dragDrop = self.findChild(QtWidgets.QWidget, 'dragdropwidget')

        self.array_len_slider = self.findChild(QtWidgets.QSlider, 'horizontalSlider')
        self.lct_slider = self.findChild(QtWidgets.QSlider, 'horizontalSlider_2')

        self.array_len_label = self.findChild(QtWidgets.QLabel, 'label')
        self.lct_label = self.findChild(QtWidgets.QLabel, 'label_2')
        self.file_label = self.findChild(QtWidgets.QLabel, 'label_3')

        self.dragDrop.setAcceptDrops(True)
        self.file_button = self.dragDrop.findChild(QtWidgets.QPushButton, 'pushButton_2')
        self.file_button.clicked.connect(self.getFile)

        self.sim_button = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.sim_button.clicked.connect(self.run_sim)

        self.array_len_slider.valueChanged.connect(self.show_array_len)
        self.lct_slider.valueChanged.connect(self.show_lct)

        self.save_video_box = self.findChild(QtWidgets.QCheckBox, 'checkBox')
        self.create_csv_box = self.findChild(QtWidgets.QCheckBox, 'checkBox_2')
        self.play_video_box = self.findChild(QtWidgets.QCheckBox, 'checkBox_3')

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

    def show_array_len(self):
        self.array_len_label.setText(str(self.array_len_slider.value()))

    def show_lct(self):
        self.lct_label.setText(str(float(self.lct_slider.value())/100))

    def run_sim(self):
        if len(self.filepaths) == 0:
            return
        if torch.cuda.is_available():
            model_path = "ai_models/model_gpu.pth"
        else:
            model_path = "ai_models/model_cpu.pth"
        for file in self.filepaths:
            try:
                video_name = glob.glob(f"{file}/*.h264")[0]
            except FileNotFoundError:
                return
            model = torch.load(model_path)

            larvae_tracker = LarvaeTracker(model, file, file,
                                           csv_write= self.create_csv_box.isChecked())
            start_time = t()
            larvae_tracker.track_video(video_name,
                                       array_len=self.array_len_slider.value(),
                                       accuracy=float(self.lct_slider.value())/100,
                                       save_video= self.save_video_box.isChecked())
            end_time = t()
            if self.play_video_box.isChecked():
                self.play(larvae_tracker.video_path)
            del larvae_tracker
            self.filepaths.pop(0)
            self.file_label.setText(f"File Names = {str([os.path.split(u)[-1] for u in self.filepaths])}")
            print(f"Entire program run time: {end_time-start_time}")

    def play(self, video):
        os.startfile(video)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec()


