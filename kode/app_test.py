"""
Some tests for the GUI
"""
import unittest
import sys
from PyQt5.QtWidgets import QApplication
from app import SliderWidget, FilterWidget

# Needed in order to create the widgets
app = QApplication(sys.argv)


class SliderWidgetTest(unittest.TestCase):
    """
    Tests the basic behavior of a SliderWidget
    """

    def value_did_change_call(self):
        """
        Notify of a change in the slider
        """
        self.value_did_change_was_called = True

    def setUp(self):
        """
        Set up the tests
        """
        self.value_did_change_was_called = False
        self.filter_widget = SliderWidget(self.value_did_change_call)

    def test_filter_widget(self):
        """
        Tests if the slider notify on change
        """
        self.value_did_change_was_called = False

        self.assertEqual(self.value_did_change_was_called, False)
        self.assertEqual(self.filter_widget.value(), 1)

        self.filter_widget.increase_effect()
        self.assertEqual(self.value_did_change_was_called, True)
        self.assertEqual(self.filter_widget.value(), 1.01)

        self.value_did_change_was_called = False
        self.filter_widget.decrease_effect()
        self.filter_widget.decrease_effect()
        self.assertEqual(self.filter_widget.value(), 0.99)
        self.assertEqual(self.value_did_change_was_called, True)


class FilterWidgetTest(unittest.TestCase):
    """
    Tests the basic behavior for a FilterWidget
    """

    def value_did_change_call(self):
        """
        Notify of a change in the slider
        """
        self.value_did_change_was_called = True

    def filter_will_be_removed(self, filter_widget):
        """
        Notify when to remove a filter
        :param filter_widget: The filter to remove
        """
        self.filter_removed_was_called = True
        self.removed_filter = filter_widget

    def setUp(self):
        """
        Set up the tests
        """
        self.value_did_change_was_called = False
        self.filter_removed_was_called = False
        self.removed_filter = None
        self.filter_widget = FilterWidget(self.value_did_change_call, self.filter_will_be_removed)

    def test_filter_widget(self):
        """
        Tests if the filter notify on change, and when the remove button is clicked
        """
        self.value_did_change_was_called = False

        self.assertEqual(self.value_did_change_was_called, False)
        self.assertEqual(self.filter_removed_was_called, False)
        self.assertEqual(self.filter_widget.effect_slider.isEnabled(), False)
        self.assertEqual(self.filter_widget.effect_value(), 1)

        self.filter_widget.effect_slider.setEnabled(True)

        self.filter_widget.effect_slider.increase_effect()
        self.assertEqual(self.value_did_change_was_called, True)
        self.assertEqual(self.filter_widget.effect_value(), 1.01)

        self.value_did_change_was_called = False
        self.filter_widget.effect_slider.decrease_effect()
        self.filter_widget.effect_slider.decrease_effect()
        self.assertEqual(self.filter_widget.effect_value(), 0.99)
        self.assertEqual(self.value_did_change_was_called, True)

        self.filter_widget.remove_button.click()
        self.assertEqual(self.filter_removed_was_called, True)
        self.assertEqual(self.removed_filter, self.filter_widget)

if __name__ == '__main__':
    unittest.main()