# gemini_krita.py
# Krita plugin: Gemini Image Editor (active-layer edit, undoable)
# Requirements: Krita (pykrita), PyQt5
# Place in Krita's pykrita/pykrita folder and restart Krita.

from krita import Krita, Extension
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice, Qt
import base64, json, tempfile, os, urllib.request, urllib.error

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"

# ----- Dialog -----
class GeminiDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gemini Image Editor")
        self.resize(520, 420)

        layout = QVBoxLayout()

        self.api_label = QLabel("API Key:")
        self.api_entry = QLineEdit()
        self.api_entry.setEchoMode(QLineEdit.Password)

        self.prompt_label = QLabel("Prompt:")
        self.prompt_entry = QLineEdit()

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)

        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        layout.addWidget(self.api_label)
        layout.addWidget(self.api_entry)
        layout.addWidget(self.prompt_label)
        layout.addWidget(self.prompt_entry)
        layout.addWidget(QLabel("Chat / Status:"))
        layout.addWidget(self.chat_view)
        layout.addWidget(self.ok_button)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def append_chat(self, text):
        self.chat_view.append(text)

    def get_api_key(self):
        return self.api_entry.text().strip()

    def get_prompt(self):
        return self.prompt_entry.text().strip()

# ----- Utils: convert raw pixels -> PNG bytes -----
def raw_rgba_to_png_bytes(raw_bytes, w, h):
    # Try to create QImage from raw bytes (RGBA8888). If not available fallback.
    # raw_bytes is Python bytes or QByteArray; ensure bytes
    if isinstance(raw_bytes, QByteArray):
        raw = bytes(raw_bytes)
    else:
        raw = raw_bytes

    # If the data already looks like PNG, return it
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw

    # bytesPerLine = width * 4 (RGBA)
    bytes_per_line = w * 4

    # Try Format_RGBA8888 (Qt >= 5.5). Fallback to ARGB32 if needed.
    qimg = QImage(raw, w, h, bytes_per_line, QImage.Format_RGBA8888)
    if qimg.isNull():
        qimg = QImage(raw, w, h, bytes_per_line, QImage.Format_ARGB32)
    if qimg.isNull():
        raise RuntimeError("Could not create QImage from node.pixelData — unsupported format")

    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
    # Save as PNG to buffer
    ok = qimg.save(buffer, "PNG")
    buffer.seek(0)
    if not ok:
        raise RuntimeError("Failed to save QImage to PNG bytes")
    png_bytes = buffer.data().data() if hasattr(buffer.data(), "data") else bytes(buffer.data())
    buffer.close()
    return png_bytes

# ----- Gemini call -----
def call_gemini(api_key, prompt, png_bytes):
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(png_bytes).decode("utf-8")
                    }}
                ]
            }
        ]
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    })

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            resp_data = json.loads(raw)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")

    # Tolerant extraction of image bytes
    img_b64 = None
    try:
        parts = resp_data["candidates"][0]["content"]["parts"]
        for p in parts:
            if "inline_data" in p and p["inline_data"].get("data"):
                img_b64 = p["inline_data"]["data"]
                break
            if "inlineData" in p and p["inlineData"].get("data"):
                img_b64 = p["inlineData"]["data"]
                break
    except Exception:
        raise RuntimeError("Unexpected Gemini response format")

    if not img_b64:
        # If there was textual feedback, include it
        text_resp = None
        for p in parts:
            if "text" in p and p["text"]:
                text_resp = p["text"]; break
        raise RuntimeError("Gemini returned no image. " + (text_resp or ""))

    return base64.b64decode(img_b64)


def png_bytes_to_raw_rgba(png_bytes, w, h):
    """Load PNG bytes into a QImage, convert/scale to (w,h) and return raw RGBA8888 bytes.

    Returns Python bytes of length w*h*4 suitable for node.setPixelData.
    """
    # Load image from data
    qimg = QImage.fromData(QByteArray(png_bytes), "PNG") if hasattr(QImage, 'fromData') else QImage()
    if qimg.isNull():
        # try alternative load
        qimg = QImage()
        if not qimg.loadFromData(png_bytes, "PNG"):
            raise RuntimeError("Failed to load PNG image data from Gemini response")

    # Ensure correct size
    if qimg.width() != w or qimg.height() != h:
        qimg = qimg.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    # Convert to RGBA8888 if available
    try:
        if qimg.format() != QImage.Format_RGBA8888:
            qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    except Exception:
        # convertToFormat may not exist or the format may differ; try ARGB32 as fallback
        try:
            if qimg.format() != QImage.Format_ARGB32:
                qimg = qimg.convertToFormat(QImage.Format_ARGB32)
            # convert ARGB32 -> RGBA ordering
            qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
        except Exception:
            pass

    # Extract raw bytes
    byte_count = qimg.byteCount()
    try:
        ptr = qimg.bits()
        # sip.voidptr supports asstring()
        raw = ptr.asstring(byte_count)
    except Exception:
        # Fallback: try to build from QImage.save into a raw buffer (slow)
        buf = QBuffer()
        buf.open(QIODevice.ReadWrite)
        if not qimg.save(buf, "RAW"):
            # RAW may not be supported; as last resort, raise
            raise RuntimeError("Unable to extract raw pixel bytes from QImage")
        raw = bytes(buf.data())
        buf.close()

    # Ensure we have expected length (w*h*4)
    expected = w * h * 4
    if len(raw) < expected:
        raise RuntimeError(f"Not enough pixel data after conversion: got {len(raw)}, expected {expected}")
    if len(raw) > expected:
        raw = raw[:expected]
    return raw


# ----- Compatibility undo/transaction context -----
class _UndoContext:
    """Context manager that attempts multiple Krita undo/transaction APIs.

    It tries, in order:
    - Document.createCommand() returning an object with commit()/rollback()
    - Document.openTransaction()
    - Document.startCommand()/endCommand() or beginCommand()/endCommand()
    If none are available it acts as a no-op context and the change will not be grouped
    in a single undo step (still applied).
    """
    def __init__(self, doc, name):
        self.doc = doc
        self.name = name
        self._mode = None
        self._obj = None

    def __enter__(self):
        d = self.doc
        # try createCommand() which some bindings provide
        try:
            if hasattr(d, 'createCommand'):
                self._obj = d.createCommand(self.name)
                self._mode = 'createCommand'
                return self
        except Exception:
            pass

        # try old openTransaction()
        try:
            if hasattr(d, 'openTransaction'):
                self._obj = d.openTransaction(self.name)
                self._mode = 'openTransaction'
                return self
        except Exception:
            pass

        # try start/begin / end style
        try:
            if hasattr(d, 'startCommand'):
                d.startCommand(self.name)
                self._mode = 'startCommand'
                return self
            if hasattr(d, 'beginCommand'):
                d.beginCommand(self.name)
                self._mode = 'beginCommand'
                return self
        except Exception:
            pass

        # no supported API found — fall back to no-op
        self._mode = None
        return self

    def __exit__(self, exc_type, exc, tb):
        d = self.doc
        # commit / rollback depending on mode
        try:
            if self._mode == 'createCommand' and self._obj is not None:
                if exc_type is None:
                    try:
                        self._obj.commit()
                    except Exception:
                        # some bindings expose commit as apply()
                        try:
                            self._obj.apply()
                        except Exception:
                            pass
                else:
                    try:
                        self._obj.rollback()
                    except Exception:
                        pass
            elif self._mode == 'openTransaction' and self._obj is not None:
                # openTransaction returned an object with commit()/rollback()
                if exc_type is None:
                    try:
                        self._obj.commit()
                    except Exception:
                        pass
                else:
                    try:
                        self._obj.rollback()
                    except Exception:
                        pass
            elif self._mode in ('startCommand', 'beginCommand'):
                # try various end/finish names
                for end_name in ('endCommand', 'finishCommand', 'commitCommand', 'endTransaction'):
                    try:
                        if hasattr(d, end_name):
                            getattr(d, end_name)()
                            break
                    except Exception:
                        continue
        except Exception:
            # never let undo management crash the plugin
            return False
        return False

# ----- Krita extension -----
class GeminiExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        action = window.createAction("gemini_image_editor", "Gemini Image Editor...", "tools/scripts")
        action.triggered.connect(lambda: self.run(window))

    def run(self, window):
        app = Krita.instance()
        doc = app.activeDocument()
        if not doc:
            QMessageBox.warning(None, "Gemini", "No active document.")
            return

        node = doc.activeNode()
        if not node:
            QMessageBox.warning(None, "Gemini", "No active layer selected.")
            return

        dlg = GeminiDialog()
        dlg.append_chat("Ready. The plugin will send the active layer only.")
        if dlg.exec_() != dlg.Accepted:
            return

        api_key = dlg.get_api_key()
        prompt = dlg.get_prompt()
        if not api_key:
            QMessageBox.warning(None, "Gemini", "API key is required.")
            return
        if not prompt:
            QMessageBox.warning(None, "Gemini", "Prompt cannot be empty.")
            return

        try:
            # Get node pixel raw bytes (for the full layer area)
            w = doc.width()
            h = doc.height()
            dlg.append_chat("Exporting active layer pixels...")
            raw = node.pixelData(0, 0, w, h)
            # Convert raw RGBA bytes -> PNG bytes
            png_in = raw_rgba_to_png_bytes(raw, w, h)

            dlg.append_chat("Contacting Gemini...")
            out_png = call_gemini(api_key, prompt, png_in)
            dlg.append_chat("Received image from Gemini — applying to active layer...")

            # Apply inside a single undoable transaction (compat across Krita versions)
            # Convert returned PNG bytes into raw RGBA bytes sized to document
            raw_out = png_bytes_to_raw_rgba(out_png, w, h)
            expected = w * h * 4
            dlg.append_chat(f"Prepared output bytes: {len(raw_out)} (expected {expected})")
            if len(raw_out) != expected:
                raise RuntimeError(f"Converted pixel buffer has wrong size: {len(raw_out)} != {expected}")

            with _UndoContext(doc, "Gemini Image Edit"):
                qba = QByteArray(raw_out)
                node.setPixelData(qba, 0, 0, w, h)
                doc.refreshProjection()

            dlg.append_chat("Done. Use Ctrl+Z to undo.")
        except Exception as e:
            dlg.append_chat("Error: " + str(e))
            QMessageBox.critical(None, "Gemini Error", str(e))
        finally:
            dlg.close()

# Register the extension
app = Krita.instance()
extension = GeminiExtension(app)
app.addExtension(extension)
