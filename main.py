#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TLM V2.0 - 传输线模型 EIS 参数辨识工具
Transmission Line Model - EIS Parameter Identification Tool V2.0
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Must set matplotlib backend before any other imports
import matplotlib
matplotlib.use('Qt5Agg')

# Add parent to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
