import os
import sys
from PyQt6.QtCore import QDir, QUrl
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QHBoxLayout
from video_visualiser import video_visualiser


class MainWidget(QMainWindow):
    OpenFile = 0
    OpenFiles = 1
    OpenDirectory = 2
    SaveFile = 3
    def __init__(self, mode=2):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.browser_mode = mode
        self.filter_name = 'All files (*.*)'
        self.dirpath = QDir.currentPath()
        self.setWindowTitle("Drag and Drop")
        self.resize(720, 480)
        self.setAcceptDrops(True)
        self.button = QPushButton('Search')
        self.button.clicked.connect(self.getFile)
        self.setCentralWidget(self.button)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            video_path = video_visualiser(f)
            if video_path is not None:
                self.play(video_path)

    def getFile(self):
        self.filepaths = []

        if self.browser_mode == MainWidget.OpenFile:
            self.filepaths.append(QFileDialog.getOpenFileName(self, caption='Choose File',
                                                              directory=self.dirpath,
                                                              filter=self.filter_name)[0])
        elif self.browser_mode == MainWidget.OpenFiles:
            self.filepaths.extend(QFileDialog.getOpenFileNames(self, caption='Choose Files',
                                                               directory=self.dirpath,
                                                               filter=self.filter_name)[0])
        elif self.browser_mode == MainWidget.OpenDirectory:
            self.filepaths.append(QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                                   directory=self.dirpath))
        else:
            options = QFileDialog.Options()
            if sys.platform == 'darwin':
                options |= QFileDialog.DontUseNativeDialog
            self.filepaths.append(QFileDialog.getSaveFileName(self, caption='Save/Save As',
                                                              directory=self.dirpath,
                                                              filter=self.filter_name,
                                                              options=options)[0])
        if len(self.filepaths) == 0:
            return
        elif len(self.filepaths) == 1:
            video_path = video_visualiser(self.filepaths[0])
            if video_path is not None:
                self.play(video_path)
        else:
            return

    def play(self, video):
        os.startfile(video)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWidget()
    ui.show()
    sys.exit(app.exec_())

