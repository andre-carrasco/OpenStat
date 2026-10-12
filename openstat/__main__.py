import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from openstat import i18n
    from openstat.ui.main_window import MainWindow

    i18n.set_language(i18n.detect_language())
    app = QApplication(sys.argv)
    app.setApplicationName("OpenStat")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
