import sys


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QHBoxLayout, QSizePolicy, QMessageBox, \
    QPushButton, QInputDialog, QSlider, QGroupBox, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from hdr import ImageSet
import numpy as np
from globalHDR import edit


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Stimat'
        self.width = 640
        self.height = 400
        self.main_image = PlotCanvas(width=5, height=4)
        self.original_image_set = ImageSet([
            ("../eksempelbilder/Balls/Balls_", "00512"),
        ])
        self.edited_image = None
        self.selected_filter_index = 0
        self.filter_options = ["ingen", "e", "ln", "pow", "sqrt", "gamma"]
        self.effect_slider = QSlider(Qt.Horizontal)
        self.init_ui()

    def init_ui(self):
        self.setup_image()
        self.setup_slider()

        group_box = QGroupBox("Hey")

        edit_button = QPushButton("Rediger Filter", self)
        edit_button.clicked.connect(self.present_filter_options)

        button_layout = QVBoxLayout()
        button_layout.addWidget(edit_button)
        button_layout.setAlignment(Qt.AlignLeading)
        button_layout.addWidget(self.effect_slider)

        group_box.setLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignLeading)
        main_layout.addWidget(group_box)
        main_layout.addWidget(self.main_image)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    def setup_image(self):
        self.main_image = PlotCanvas(width=5, height=4)
        self.main_image.plot_image(self.original_image_set.images[0])

    def setup_slider(self):
        self.effect_slider = QSlider(Qt.Horizontal)
        self.effect_slider.setMaximum(4)
        self.effect_slider.setMinimum(0)
        #self.effect_slider.valueChanged.connect(self.update_image_with_filter())

    def present_filter_options(self):
        filter_name, confirmed = QInputDialog.getItem(self, "Velg Filter", "Filter: ", self.filter_options,\
                                                      self.selected_filter_index, False)

        new_index = self.filter_options.index(filter_name)
        if confirmed and self.selected_filter_index != new_index and filter_name != "ingen":
            self.selected_filter_index = new_index
            self.update_image_with_filter()

    def update_image_with_filter(self):
        filter_name = self.filter_options[self.selected_filter_index]
        self.edited_image = self.original_image_set.images[0] / 255

        effect_value = self.effect_slider.value()
        print(effect_value)

        if filter_name != "ingen":
            self.edited_image = edit(self.edited_image, filter_name)

        self.edited_image[self.edited_image > 1] = 1
        self.edited_image[self.edited_image < 0] = 0
        self.main_image.plot_image(self.edited_image)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=10, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_graph(self, x_axis, y_axis, title=""):
        plt = self.figure.add_subplot(111)
        plt.plot(x_axis, y_axis)
        plt.set_title(title)
        self.draw()

    def plot_image(self, image, title=""):
        plt = self.figure.add_subplot(111)
        plt.axis("off")
        plt.imshow(image)
        plt.set_title(title)
        self.draw()

    def plot(self):
        color_images = ImageSet([
            ("../eksempelbilder/Balls/Balls_", "00001"),
            ("../eksempelbilder/Balls/Balls_", "00004"),
            ("../eksempelbilder/Balls/Balls_", "00016"),
            ("../eksempelbilder/Balls/Balls_", "00032"),
            ("../eksempelbilder/Balls/Balls_", "00064"),
            ("../eksempelbilder/Balls/Balls_", "00128"),
            ("../eksempelbilder/Balls/Balls_", "00256"),
            ("../eksempelbilder/Balls/Balls_", "00512"),
            ("../eksempelbilder/Balls/Balls_", "01024"),
            ("../eksempelbilder/Balls/Balls_", "02048"),
            # ("../eksempelbilder/Balls/Balls_", "04096"),
            # ("../eksempelbilder/Balls/Balls_", "08192"),
            # ("../eksempelbilder/Balls/Balls_", "16384"),
            # load_image("../eksempelbilder/Balls/Balls_", "01024"),
            # load_image("../eksempelbilder/Balls/Balls_", "02048"),
        ])
        color_hrd_map = color_images.hdr(10)
        z_values = np.arange(0, 256)
        ax = self.figure.add_subplot(111)
        ax.plot(color_hrd_map[0][0], z_values)
        ax.set_title('PyQt Matplotlib Example.')
        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
