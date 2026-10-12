"""Output pane: renders the project's output history as one HTML document."""
from __future__ import annotations

import base64

from PySide6.QtGui import QImage, QTextDocument
from PySide6.QtWidgets import QTextBrowser

from openstat.core.result import wrap_document


class OutputView(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)

    def loadResource(self, type_, url):
        # Figures are embedded as base64 data URLs
        if type_ == QTextDocument.ImageResource and url.scheme() == "data":
            payload = url.toString().split(",", 1)[-1]
            image = QImage()
            image.loadFromData(base64.b64decode(payload))
            return image
        return super().loadResource(type_, url)

    def show_outputs(self, outputs, scroll_to: int | None = None) -> None:
        self.setHtml(wrap_document("\n<hr>\n".join(o.html for o in outputs)))
        if scroll_to is not None:
            self.scrollToAnchor(f"r{scroll_to}")
