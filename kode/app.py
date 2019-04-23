"""
A module that presents a GUI
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QSizePolicy, \
    QPushButton, QInputDialog, QSlider, QGroupBox, QWidget, QLabel, QFileDialog, QScrollArea

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from globalHDR import edit_luminance, edit_globally, read_image
from image_set import ImageSet


class App(QWidget):
    """
    A widget representing the main app GUI
    """

    def __init__(self):
        super().__init__()
        self.title = 'Stimat'
        self.main_image = PlotCanvas(width=5, height=4)
        self.original_image_set = ImageSet([])
        self.edited_image = None
        self.hdr_image = None
        self.filter_widgets = list()
        self.filter_layout = QVBoxLayout()
        self.add_global_filter_button = QPushButton("Legg til globalt filter", self)
        self.add_lum_filter_button = QPushButton("Legg til luminans filter", self)
        self.status_label = QLabel("Ingen bilder er lastet inn")
        self.init_ui()

    def init_ui(self):
        """
        Setup all the ui in the widget
        """
        self.setup_image()
        self.filter_layout = QVBoxLayout()

        open_image_button = QPushButton("Last inn bilde (velg flere for HDR-rekonstruksjon)", self)
        open_image_button.clicked.connect(self.select_file)

        self.add_global_filter_button.clicked.connect(self.add_global_filter)
        self.add_global_filter_button.setEnabled(False)

        self.add_lum_filter_button.clicked.connect(self.add_lum_filter)
        self.add_lum_filter_button.setEnabled(False)

        self.status_label.setStyleSheet("background: orange")
        self.status_label.setContentsMargins(10, 3, 10, 3)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.status_label)
        button_layout.addWidget(open_image_button)
        button_layout.addWidget(self.add_global_filter_button)
        button_layout.addWidget(self.add_lum_filter_button)
        button_layout.setAlignment(Qt.AlignCenter)

        group_box = QGroupBox("Instillinger")
        group_box.setLayout(button_layout)
        group_box.setMaximumWidth(400)

        self.filter_layout.addStretch()

        filter_box = QGroupBox("Filter")
        filter_box.setLayout(self.filter_layout)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(filter_box)
        scroll.setMaximumWidth(400)

        action_layout = QVBoxLayout()
        action_layout.addWidget(group_box)
        action_layout.addWidget(scroll)

        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignLeading)
        main_layout.addWidget(self.main_image)
        main_layout.addLayout(action_layout)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.show()

    def setup_image(self):
        """
        Setup the image GUI
        """
        self.main_image = PlotCanvas(width=5, height=4)

    def add_global_filter(self):
        """
        Adds a filter to the layout
        """
        new_filter = FilterWidget(self.update_image_with_filter, self.remove_filter)
        self.filter_widgets.append(new_filter)
        self.filter_layout.addWidget(new_filter)

    def add_lum_filter(self):
        """
        Adds a filter to the layout
        """
        new_filter = LumimanceFilterWidget(self.update_image_with_filter, self.remove_filter)
        self.filter_widgets.append(new_filter)
        self.filter_layout.addWidget(new_filter)

    def remove_filter(self, filter_widget):
        """
        Removes a filter widget

        :param filter_widget: The filter widget to remove
        :type filter_widget: QWidget
        """
        filter_widget.deleteLater()
        self.filter_layout.removeWidget(filter_widget)
        self.filter_widgets.remove(filter_widget)
        self.update_image_with_filter()

    def update_image_with_filter(self):
        """
        Updates the image with the selected filter and effects
        """
        self.edited_image = self.original_image_set.images[0].copy() / 255
        if self.hdr_image is not None:
            self.edited_image = self.hdr_image.copy()

        for filter_widget in self.filter_widgets:
            self.edited_image = filter_widget.apply_filter(self.edited_image)

        scaled_image = (self.edited_image - self.edited_image.min())/(self.edited_image.max() - self.edited_image.min())
        self.main_image.plot_image(scaled_image)

    def select_file(self):
        """
        Selects a set of files and generates a HDR image
        """
        file_name, ok = QFileDialog.getOpenFileNames(self, "Velg bilde", "", "PNG (*.png);;EXR (*.exr)")

        if ok:
            self.add_global_filter_button.setEnabled(True)
            self.add_lum_filter_button.setEnabled(True)

            try:
                if file_name[0].endswith(".exr"):
                    self.original_image_set = None
                    self.hdr_image = read_image(file_name)
                else:
                    image_info = list(map(lambda file: (file, file.rsplit("_", 1)[-1].replace(".png", "")), file_name))
                    self.original_image_set = ImageSet(image_info).aligned_image_set()
                    self.hdr_image = self.original_image_set.hdr_image(10)

                self.update_image_with_filter()
                self.status_label.setText("Bilde ble lastet inn")
                self.status_label.setStyleSheet("background: green")
            except:
                self.status_label.setText("Ups! Det skjedde en feil ved innlasting av bildet")
                self.status_label.setStyleSheet("background: red")


class SliderWidget(QWidget):
    """
    A Widget that presents a filter setting
    """

    def __init__(self, value_did_change_function, title=None, parent=None):
        super(SliderWidget, self).__init__(parent)
        self.effect_slider = QSlider(Qt.Horizontal)
        self.effect_value_label = QLabel("1")
        self.value_did_change_function = value_did_change_function
        self.title = title
        self.init_ui()

    def init_ui(self):
        """
        Inits the widget ui
        """

        vertical = QVBoxLayout()
        vertical.setSpacing(8)
        vertical.setContentsMargins(12, 0, 0, 0)

        if self.title is not None:
            label = QLabel(self.title)
            vertical.addWidget(label)

        slider_box = QHBoxLayout()
        dec_button = QPushButton("-")
        dec_button.clicked.connect(self.decrease_effect)

        inc_button = QPushButton("+")
        inc_button.clicked.connect(self.increase_effect)

        self.effect_value_label = QLabel("1")
        self.effect_value_label.setMaximumWidth(30)
        self.effect_value_label.setMinimumWidth(30)

        self.setup_slider()

        slider_box.addWidget(self.effect_value_label)
        slider_box.addWidget(dec_button)
        slider_box.addWidget(inc_button)
        slider_box.addWidget(self.effect_slider)

        vertical.addLayout(slider_box)

        self.setLayout(vertical)
        self.show()

    def setup_slider(self):
        """
        Setup the slider GUI
        """
        self.effect_slider = QSlider(Qt.Horizontal)
        self.effect_slider.setMaximum(400)
        self.effect_slider.setMinimum(0)
        self.effect_slider.setValue(100)
        self.effect_slider.setMinimum(1)
        self.effect_slider.valueChanged.connect(self.slider_did_change)

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

    def value(self):
        """
        :return: The value of the slider
        """
        return self.effect_slider.value() / 100

    def slider_did_change(self):
        """
        A function that is called when the slider change value
        """
        effect_value = self.value()
        self.effect_value_label.setText(str(effect_value))
        self.value_did_change_function()


class FilterWidget(QWidget):
    """
    A Widget that presents a filter setting
    """

    def __init__(self, value_did_change_function, remove_filter_function, parent=None):
        super(FilterWidget, self).__init__(parent)
        self.selected_filter_index = 0
        self.filter_options = ["ingen", "e", "ln", "pow", "sqrt", "gamma"]
        self.selected_filter_label = QLabel("Valgt effekt: Ingen")
        self.effect_slider = QSlider(Qt.Horizontal)
        self.effect_value_label = QLabel("1")
        self.effect_label = QLabel("Effekt: Ingen")
        self.remove_button = QPushButton("Slett", self)
        self.remove_filter_function = remove_filter_function
        self.value_did_change_function = value_did_change_function
        self.effect_slider = SliderWidget(value_did_change_function, "Effekt styrke")
        self.effect_layout = QVBoxLayout(self)
        self.init_ui()

    def init_ui(self):
        """
        Creates the UI for the widget
        """

        filter_box = QHBoxLayout()

        edit_button = QPushButton("Rediger Filter", self)
        edit_button.clicked.connect(self.present_filter_options)

        self.remove_button.clicked.connect(self.remove_was_clicked)

        filter_box.addWidget(self.effect_label)
        filter_box.addWidget(edit_button)
        filter_box.addWidget(self.remove_button)

        self.effect_slider.setEnabled(False)

        self.effect_layout.addLayout(filter_box)
        self.effect_layout.addWidget(self.effect_slider)
        self.effect_layout.setSpacing(4)

        self.setLayout(self.effect_layout)
        self.show()

    def effect_value(self):
        """
        Calculates the slider value

        :return: The value of the slider
        """
        return self.effect_slider.value()

    def present_filter_options(self):
        """
        Presents the different filter options
        """
        filter_name, confirmed = QInputDialog.getItem(self, "Velg Filter", "Filter: ", self.filter_options,\
                                                      self.selected_filter_index, False)
        new_index = self.filter_options.index(filter_name)

        if confirmed and self.selected_filter_index != new_index:
            self.effect_label.setText("Effekt: " + filter_name)
            self.selected_filter_label.setText("Valgt effekt: " + filter_name)
            self.selected_filter_index = new_index

            if filter_name in ("pow", "gamma"):
                self.effect_slider.setEnabled(True)
            else:
                self.effect_slider.setEnabled(False)

            self.value_did_change_function()

    def remove_was_clicked(self):
        """
        A function that will be called when the remove button is clicked
        """
        self.remove_filter_function(self)

    def apply_filter(self, image):
        """
        Apply a filter to a image

        :param image: The image to apply the filter on
        :type image: Numpy array

        :return: A new image with the filter on
        """
        filter_name = self.filter_options[self.selected_filter_index]
        if filter_name != "ingen":
            effect_value = self.effect_value()
            return edit_globally(image, effect=effect_value, func=filter_name)
        else:
            return image


class LumimanceFilterWidget(FilterWidget):
    """
    A Widget that presents a filter setting
    """

    def __init__(self, value_did_change_function, remove_filter_function, parent=None):
        super(LumimanceFilterWidget, self).__init__(value_did_change_function, remove_filter_function, parent)
        self.luminance_slider = SliderWidget(value_did_change_function, "Luminance")
        self.chromasity_slider = SliderWidget(value_did_change_function, "Chromasity")
        self.effect_layout.addWidget(self.luminance_slider)
        self.effect_layout.addWidget(self.chromasity_slider)

    def luminance_value(self):
        """
        :return: The luminance value
        """
        return self.luminance_slider.value()

    def chromasity_value(self):
        """
        :return: The chromasity value
        """
        return self.chromasity_slider.value()

    def apply_filter(self, image):
        """
        Apply a filter to a image

        :param image: The image to apply the filter on
        :type image: Numpy array

        :return: A new image with the filter on
        """
        filter_name = self.filter_options[self.selected_filter_index]
        if filter_name != "ingen":
            return edit_luminance(image, self.effect_value(), self.luminance_value(),
                                  self.chromasity_value(), filter_name)
        else:
            return image


class PlotCanvas(FigureCanvas):
    """
    A Widget view that presents a plot or image
    """

    current_figure = None
    axes = None

    def __init__(self, parent=None, width=10, height=10, dpi=100):
        self.current_figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.current_figure.add_subplot(111)

        FigureCanvas.__init__(self, self.current_figure)
        self.current_figure.patch.set_facecolor("none")
        self.setStyleSheet("background-color:rgba(0, 0, 0, 0)")
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
        :type x_axis: Numpy array

        :param y_axis: The y-axis to display
        :type y_axis: Numpy array

        :param title: The title to display
        :type title: str
        """
        self.clear_figure()
        self.current_figure = self.figure.add_subplot(111)
        self.current_figure.plot(x_axis, y_axis)
        self.current_figure.set_title(title)
        self.draw()

    def plot_image(self, image, title=""):
        """
        Displays a image in a view

        :param image: The image to display
        :type image: Numpy array

        :param title: The title to display
        :type title: str
        """
        self.clear_figure()
        self.current_figure = self.figure.add_subplot(111)
        self.current_figure.axis("off")
        if image.ndim == 2:
            self.current_figure.imshow(image, plt.cm.gray)
        else:
            self.current_figure.imshow(image)
        self.current_figure.set_title(title)
        self.draw()

    def clear_figure(self):
        """
        Clears the current figure
        """
        if self.current_figure is not None:
            plt.close(111)
            self.current_figure = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
