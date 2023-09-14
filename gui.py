from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QCheckBox, QPlainTextEdit, QPushButton

class SortingApp(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # 1. Text box with padding
        self.text_box = QTextEdit(self)
        self.text_box.setPlaceholderText("Enter your text here...")
        layout.addWidget(self.text_box)

        # 2. List of checkboxes with sorting algorithms
        self.checkboxes = {}
        from main import all_algorithms
        for algo in all_algorithms:
            checkbox = QCheckBox(algo["name"], self)
            self.checkboxes[algo["id"]] = checkbox
            layout.addWidget(checkbox)

        # Button that will run a function when clicked
        self.run_button = QPushButton("Run Algorithms", self)
        layout.addWidget(self.run_button)

        self.analyze = QPushButton("Analyze", self)
        layout.addWidget(self.analyze)

        self.show_stats = QPushButton("Show Stats", self)
        layout.addWidget(self.show_stats)

        # 3. White box for logs
        self.log_box = QPlainTextEdit(self)
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)
        self.setWindowTitle("Sorting App")

        # Set the window size
        self.resize(600, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.show()

    def sample_function(self):
        selected_algorithms = self.get_selected_checkboxes()
        self.log_message(f"Selected algorithms: {', '.join(selected_algorithms)}")

    def log_message(self, message):
        """Function to display a message in the log box."""
        self.log_box.appendPlainText(message)

    # Utility functions

    def get_checkboxes(self):
        return self.checkboxes

    def update_checkboxes(self, selected_algorithms):
        for algo, checkbox in self.checkboxes.items():
            checkbox.setChecked(algo in selected_algorithms)

    def update_input_text(self, text):
        self.text_box.setPlainText(text)

    def debug(self, message):
        """Function to print a debug message to the terminal."""
        print(f"DEBUG: {message}")

    def get_input_text(self):
        return self.text_box.toPlainText()

    def on_run_button_clicked(self, callback):
        self.run_button.clicked.connect(callback)

    def on_analyze_button_clicked(self, callback):
        self.analyze.clicked.connect(callback)

    def on_show_stats_button_clicked(self, callback):
        self.show_stats.clicked.connect(callback)
