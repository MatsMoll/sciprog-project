"""
A module that presents a GUI
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QHBoxLayout, QSizePolicy, QMessageBox, \
    QPushButton, QInputDialog, QSlider, QGroupBox, QWidget, QLabel

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from hdr import ImageSet
#import numpy as np
from globalHDR import split_image, read_image


class App(QWidget):
    """
    A widget representing the main app GUI
    """

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Stimat'
        self.width = 1000
        self.height = 600
        self.main_image = PlotCanvas(width=5, height=4)
        self.original_image_set = ImageSet(
            [("../eksempelbilder/Balls/Balls_", "01024")]
        )
        self.edited_image = None
        self.selected_filter_index = 0
        self.filter_options = ["ingen", "e", "ln", "pow", "sqrt", "gamma"]
        self.selected_filter_label = QLabel("Valgt effekt: Ingen")
        self.effect_slider = QSlider(Qt.Horizontal)
        self.effect_label = QLabel("1")
        self.init_ui()

    def init_ui(self):
        """
        Setup all the ui in the widget
        """
        self.setup_image()
        self.setup_slider()

        group_box = QGroupBox("Instillinger")

        edit_button = QPushButton("Rediger Filter", self)
        edit_button.clicked.connect(self.present_filter_options)

        effect_group = QVBoxLayout()
        effect_label = QLabel("Effekt")
        slider_box = QHBoxLayout()

        dec_button = QPushButton("-")
        dec_button.clicked.connect(self.decrease_effect)

        inc_button = QPushButton("+")
        inc_button.clicked.connect(self.increase_effect)

        self.effect_label = QLabel("0.01")
        self.effect_label.setMaximumWidth(30)
        self.effect_label.setMinimumWidth(30)

        slider_box.addWidget(dec_button)
        slider_box.addWidget(inc_button)
        slider_box.addWidget(self.effect_label)
        slider_box.addWidget(self.effect_slider)

        effect_group.addWidget(effect_label)
        effect_group.addLayout(slider_box)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.selected_filter_label)
        button_layout.addWidget(edit_button)
        button_layout.addStretch()
        button_layout.addLayout(effect_group)
        button_layout.setAlignment(Qt.AlignLeading)

        group_box.setLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignLeading)
        main_layout.addWidget(group_box)
        main_layout.addWidget(self.main_image)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()
        self.update_image_with_filter()

    def setup_image(self):
        """
        Setup the image GUI
        """
        self.main_image = PlotCanvas(width=5, height=4)
        self.main_image.plot_image(self.original_image_set.images[0])

    def setup_slider(self):
        """
        Setup the slider GUI
        """
        self.effect_slider = QSlider(Qt.Horizontal)
        self.effect_slider.setMaximum(400)
        self.effect_slider.setMinimum(0)
        self.effect_slider.setValue(1)
        self.effect_slider.valueChanged.connect(self.update_image_with_filter)

    def decrease_effect(self):
        """
        Decrease the effect slider by one step
        """
        if self.effect_slider.isEnabled():
            self.effect_slider.setValue(self.effect_slider.value() - 1)

    def increase_effect(self):
        """
        Increase the effect slider by one step
        """
        if self.effect_slider.isEnabled():
            self.effect_slider.setValue(self.effect_slider.value() + 1)

    def present_filter_options(self):
        """
        Presents the different filter options
        """
        filter_name, confirmed = QInputDialog.getItem(self, "Velg Filter", "Filter: ", self.filter_options,\
                                                      self.selected_filter_index, False)

        new_index = self.filter_options.index(filter_name)

        if confirmed and self.selected_filter_index != new_index:
            self.selected_filter_label.setText("Valgt effekt: " + filter_name)
            self.selected_filter_index = new_index

            if filter_name == "pow" or filter_name == "gamma":
                self.effect_slider.setEnabled(True)
            else:
                self.effect_slider.setEnabled(False)

            self.update_image_with_filter()

    def update_image_with_filter(self):
        """
        Updates the image with the selected filter and effects
        """
        filter_name = self.filter_options[self.selected_filter_index]
        self.edited_image = self.original_image_set.images[0] / 255

        effect_value = self.effect_slider.value() / 10
        self.effect_label.setText(str(round(effect_value * 10) / 100))

        if filter_name != "ingen":
            self.edited_image = split_image(self.edited_image, filter_name, effect_value)

        self.edited_image[self.edited_image > 1] = 1
        self.edited_image[self.edited_image < 0] = 0
        self.main_image.plot_image(self.edited_image)


class PlotCanvas(FigureCanvas):
    """
    A Widget view that presents a plot or image
    """

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
        """
        Plots a graph in the view
        :param x_axis: The x-axis to display
        :param y_axis: The y-axis to display
        :param title: The title to display
        """
        plt = self.figure.add_subplot(111)
        plt.plot(x_axis, y_axis)
        plt.set_title(title)
        self.draw()

    def plot_image(self, image, title=""):
        """
        Displays a image in a view
        :param image: The image to display
        :param title: The title to display
        """
        plt = self.figure.add_subplot(111)
        plt.axis("off")
        plt.imshow(image)
        plt.set_title(title)
        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
