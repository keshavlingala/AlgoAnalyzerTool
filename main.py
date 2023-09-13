import sys

from PyQt5.QtWidgets import QApplication

from gui import SortingApp


def run_callback(args):
    ex.log_message(f"Run button clicked from {args}")


def analyze_callback(args):
    ex.log_message(f"Analyze button clicked from {args}")


def show_stats_callback(args):
    ex.log_message(f"Show Stats button clicked from {args}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SortingApp()
    ex.on_run_button_clicked(run_callback)
    ex.on_analyze_button_clicked(analyze_callback)
    ex.on_show_stats_button_clicked(show_stats_callback)
    sys.exit(app.exec_())
