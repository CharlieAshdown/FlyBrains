from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import QDir
import sys
import os
import yaml



class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('DataCollection.ui', self) # Load the .ui file
        self.filepath = os.curdir
        self.configs = {}
        self.file_name = ""
        self.dirpath = QDir.currentPath()
        self.yaml_drop = self.findChild(QtWidgets.QWidget, 'yaml_input')
        self.warning = self.findChild(QtWidgets.QLabel, 'warning')

        self.edit_yaml = self.findChild(QtWidgets.QCheckBox, 'edit_yaml')

        self.working_dir = self.findChild(QtWidgets.QLabel, 'working_dir')

        self.test_number = self.findChild(QtWidgets.QTextEdit, 'test_num')
        self.yaml_name = self.findChild(QtWidgets.QTextEdit, 'yaml_name')

        self.exposure = self.findChild(QtWidgets.QTextEdit, 'exposure')
        self.gain = self.findChild(QtWidgets.QTextEdit, 'gain')
        self.camera_delay = self.findChild(QtWidgets.QTextEdit, 'camera_delay')
        self.record_time = self.findChild(QtWidgets.QTextEdit, 'record_time')

        self.ir_frequency = self.findChild(QtWidgets.QTextEdit, 'IR_frequency')
        self.ir_duty = self.findChild(QtWidgets.QSlider, 'IR_duty')
        self.ir_duty_label = self.findChild(QtWidgets.QLabel, 'IR_duty_value')
        self.ir_duty.valueChanged.connect(self.update_ir_duty)

        self.opto_delay = self.findChild(QtWidgets.QTextEdit, 'opto_delay')
        self.opto_flash = self.findChild(QtWidgets.QTextEdit, 'opto_flash')
        self.opto_frequency = self.findChild(QtWidgets.QTextEdit, 'opto_frequency')
        self.opto_duty = self.findChild(QtWidgets.QSlider, 'opto_duty')
        self.opto_duty_label = self.findChild(QtWidgets.QLabel, 'opto_duty_value')
        self.opto_duty.valueChanged.connect(self.update_opto_duty)

        self.yaml_drop.setAcceptDrops(True)
        self.yaml_button = self.yaml_drop.findChild(QtWidgets.QPushButton, 'yaml_button')
        self.yaml_button.clicked.connect(self.getFile)

        self.sim_button = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.sim_button.clicked.connect(self.get_video)

        self.edit_yaml.clicked.connect(self.enable_write)

        self.yaml_name.textChanged.connect(self.change_warning)

        self.change_work_dir = self.findChild(QtWidgets.QPushButton, 'change_work_dir')
        self.change_work_dir.clicked.connect(self.change_directories)

        self.sim_data = self.findChild(QtWidgets.QTextEdit, 'textEdit')

        self.show()  # Show the GUI

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.filepath = [u.toLocalFile() for u in event.mimeData().urls()][0]
        self.working_dir.setText(f"Working directory = {self.filepath}")
        self.read_yaml()
        self.edit_yaml.setEnabled(True)
        self.enable_write()

    def getFile(self):
        self.filepath = (QFileDialog.getOpenFileName(self, caption='Choose Yaml file',
                                                          directory=self.dirpath,
                                                          filter='(*.yml, *.yaml)'))[0]
        self.dirpath = os.path.split(self.filepath)[0]
        self.working_dir.setText(f"Working directory = {self.dirpath}")
        self.read_yaml()
        self.edit_yaml.setEnabled(True)
        self.enable_write()

    def enable_write(self):
        if self.edit_yaml.isChecked():
            self.warning.setText('Warning: You are currently overwriting the existing simulation file')
            self.exposure.setEnabled(True)
            self.gain.setEnabled(True)
            self.camera_delay.setEnabled(True)
            self.record_time.setEnabled(True)
            self.ir_frequency.setEnabled(True)
            self.ir_duty.setEnabled(True)
            self.opto_delay.setEnabled(True)
            self.opto_flash.setEnabled(True)
            self.opto_frequency.setEnabled(True)
            self.opto_duty.setEnabled(True)
            self.yaml_name.setEnabled(True)
        else:
            self.exposure.setEnabled(False)
            self.gain.setEnabled(False)
            self.camera_delay.setEnabled(False)
            self.record_time.setEnabled(False)
            self.ir_frequency.setEnabled(False)
            self.ir_duty.setEnabled(False)
            self.opto_delay.setEnabled(False)
            self.opto_flash.setEnabled(False)
            self.opto_frequency.setEnabled(False)
            self.opto_duty.setEnabled(False)
            self.yaml_name.setEnabled(False)

    def read_yaml(self):
        with open(self.filepath, 'r') as file:
            self.configs = yaml.safe_load(file)
        self.file_name = os.path.split(os.path.splitext(self.filepath)[0])[-1]
        self.yaml_name.setText(self.file_name)

        self.exposure.setText(str(self.configs["camera"]["exposure"]))
        self.gain.setText(str(self.configs["camera"]["gain"]))
        self.camera_delay.setText(str(self.configs["camera"]["delay"]))
        self.record_time.setText(str(self.configs["camera"]["record_time"]))

        self.ir_frequency.setText(str(self.configs["IR_LED"]["frequency"]))
        self.ir_duty.setValue(int(self.configs["IR_LED"]["duty"]))

        self.opto_delay.setText(str(self.configs["Optogenetic_LEDs"]["initial_delay"]))
        self.opto_flash.setText(str(self.configs["Optogenetic_LEDs"]["flash_length"]))
        self.opto_frequency.setText(str(self.configs["Optogenetic_LEDs"]["frequency"]))
        self.opto_duty.setValue(int(self.configs["Optogenetic_LEDs"]["duty"]))

    def write_yaml(self):
        self.configs["camera"] = \
            {
                "exposure": int(self.exposure.toPlainText()),
                "gain": int(self.gain.toPlainText()),
                "delay": int(self.camera_delay.toPlainText()),
                "record_time": int(self.record_time.toPlainText())
            }
        self.configs["IR_LED"] = \
            {
                "frequency": int(self.ir_frequency.toPlainText()),
                "duty": int(self.ir_duty.value())
            }
        self.configs["Optogenetic_LEDs"] = \
            {
                "initial_delay": int(self.opto_delay.toPlainText()),
                "flash_length": int(self.opto_flash.toPlainText()),
                "frequency": int(self.opto_frequency.toPlainText()),
                "duty": int(self.opto_duty.value())
            }
        with open(f"{self.dirpath}/{self.yaml_name.toPlainText()}.yaml", 'w') as outfile:
            yaml.dump(self.configs, outfile)

    def change_warning(self):
        if self.edit_yaml.isChecked():
            if self.yaml_name.toPlainText() == self.file_name:
                self.warning.setText('Warning: You are currently overwriting the existing simulation file')
            else:
                self.warning.setText('')
        else:
            self.warning.setText('')

    def update_ir_duty(self):
        self.ir_duty_label.setText(str(self.ir_duty.value()))

    def update_opto_duty(self):
        self.opto_duty_label.setText(str(self.opto_duty.value()))

    def change_directories(self):
        self.dirpath = (QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                               directory=self.dirpath))
        self.working_dir.setText(f"Working directory = {self.dirpath}")

    def get_video(self):
        self.write_yaml()
        sim_folder = f"{os.path.split(self.filepath)[0]}/test_{self.test_number.toPlainText().zfill(3)}"
        if not os.path.exists(sim_folder):
            os.mkdir(sim_folder)
        with open(f"{sim_folder}/README.txt", 'w') as f:
            f.write(self.sim_data.toPlainText())
        #os.system(f"cd {os.curdir}")
        #os.system(f"python video.py {self.filepath} {int(self.test_number.toPlainText())}"
                  #f" & python Optogenetic_LEDs.py {self.filepath} {int(self.test_number.toPlainText())} &")
        self.filepath = os.curdir


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec()


