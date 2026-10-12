import sys


def main() -> int:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QMessageBox

    from openstat import i18n
    from openstat.core import registry
    from openstat.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("OpenStat")
    saved = QSettings("OpenStat", "OpenStat").value("language")
    i18n.set_language(saved if saved in ("pt_BR", "en") else i18n.detect_language())
    plugin_errors = registry.load_plugins()
    window = MainWindow()
    window.show()
    if plugin_errors:
        QMessageBox.warning(window, "OpenStat", i18n.tr("Some plugins could not be loaded:") + "\n\n"
                            + "\n".join(plugin_errors))
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".ostat"):
        from openstat.core.project import Project
        window._guarded(lambda: window.set_project(Project.load(sys.argv[1])))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
