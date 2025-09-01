# gemini_krita.py
# Krita plugin: Gemini Image Editor (active-layer edit, undoable)
# Requirements: Krita (pykrita), PyQt5
# Place in Krita's pykrita/pykrita folder and restart Krita.

from krita import Krita, Extension
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox, QListWidget, QListWidgetItem, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice, Qt, QSize
import base64, json, urllib.request, urllib.error, hashlib

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"

# ----- Dialog ----- 
class GeminiDialog(QDialog):
    def __init__(self, doc, node, saved_key="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gemini Image Editor")
        self.resize(760, 420)

        self.doc = doc
        self.node = node
        self.app = Krita.instance()

        # main horizontal layout: controls on left, preview queue on right
        main_h = QHBoxLayout()
        left_v = QVBoxLayout()
        right_v = QVBoxLayout()

        # --- left column: controls ---
        self.api_label = QLabel("API Key:")
        self.api_entry = QLineEdit()
        self.api_entry.setEchoMode(QLineEdit.Password)
        self.api_entry.setText(saved_key)

        self.prompt_label = QLabel("Prompt:")
        self.prompt_entry = QLineEdit()

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)

        self.ok_button = QPushButton("Generate!")
        self.cancel_button = QPushButton("Close")
        # undo/redo buttons removed; preview queue handles rollbacks
        self.ok_button.clicked.connect(self.run_gemini_processing)
        self.cancel_button.clicked.connect(self.reject)

        # plugin-level history (append-only list of raw RGBA bytes) and current index
        self._history = []
        self._current_index = -1
        # map of snapshot hash -> index in _history for quick dedup checks
        self._hash_map = {}

        left_v.addWidget(self.api_label)
        left_v.addWidget(self.api_entry)
        left_v.addWidget(self.prompt_label)
        left_v.addWidget(self.prompt_entry)
        left_v.addWidget(QLabel("Chat / Status:"))
        left_v.addWidget(self.chat_view)
        left_v.addWidget(self.ok_button)
        left_v.addWidget(self.cancel_button)
        # ...undo/redo removed

        # --- right column: preview queue ---
        self.preview_list = QListWidget()
        self.preview_list.setIconSize(QSize(160, 90))
        self.preview_list.setFixedWidth(180)
        self.preview_list.itemClicked.connect(self._on_preview_selected)
        right_v.addWidget(QLabel("Preview Queue"))
        right_v.addWidget(self.preview_list)

        main_h.addLayout(left_v)
        main_h.addLayout(right_v)

        layout = QVBoxLayout()
        layout.addLayout(main_h)

        self.setLayout(layout)
        self.append_chat("Ready. The plugin will send the active layer only.")

    def append_chat(self, text):
        self.chat_view.append(text)
        self.chat_view.repaint()

    def get_api_key(self):
        return self.api_entry.text().strip()

    def get_prompt(self):
        return self.prompt_entry.text().strip()

    # undo/redo removed in favor of preview queue



    def _apply_raw(self, raw_bytes):
        """Apply raw RGBA bytes to the node inside an undo context."""
        w = self.doc.width()
        h = self.doc.height()
        expected = w * h * 4
        if len(raw_bytes) != expected:
            raise RuntimeError(f"Snapshot buffer size mismatch: {len(raw_bytes)} != {expected}")
        with _UndoContext(self.doc, "Gemini Image Edit (plugin-level)"):
            qba = QByteArray(raw_bytes)
            self.node.setPixelData(qba, 0, 0, w, h)
            self.doc.refreshProjection()

    def _add_snapshot(self, raw_bytes, label=None, append_to_end=False):
        """Store snapshot in history and add thumbnail to preview list."""
        # Normalize to Python bytes for consistent comparisons/storage
        if isinstance(raw_bytes, QByteArray):
            raw = bytes(raw_bytes)
        else:
            raw = raw_bytes

        # If we're not at the end of the history, truncate the branch (clear redo)
        # By default we truncate when adding snapshots from user actions; however
        # if append_to_end=True we will always append to the end and not truncate
        # so prompt results won't overwrite selected previews.
        if not append_to_end and self._current_index < len(self._history) - 1:
            keep = self._current_index + 1
            # remove preview items beyond keep
            while self.preview_list.count() > keep:
                itm = self.preview_list.takeItem(self.preview_list.count() - 1)
                try:
                    del itm
                except Exception:
                    pass
            self._history = self._history[:keep]
            # rebuild hash map for truncated history
            self._hash_map = {}
            for i, hraw in enumerate(self._history):
                hh = hashlib.sha256(hraw).hexdigest()
                self._hash_map[hh] = i
            # ensure preview_list and history counts match
            while self.preview_list.count() > len(self._history):
                itm = self.preview_list.takeItem(self.preview_list.count() - 1)
                try:
                    del itm
                except Exception:
                    pass

        # Deduplicate using SHA-256 checksum across the entire history.
        new_hash = hashlib.sha256(raw).hexdigest()
        if new_hash in self._hash_map:
            # Already present somewhere in stack; set current index to existing
            self._current_index = self._hash_map[new_hash]
            try:
                self.preview_list.setCurrentRow(self._current_index)
            except Exception:
                pass
            return

        # append to history and set current index to new snapshot
        self._history.append(raw)
        # store hash map entry for quick dedup checks
        self._hash_map[new_hash] = len(self._history) - 1
        # If append_to_end=True we want the new snapshot to be at the logical end
        # of the stack, so set current index to the new last element.
        self._current_index = len(self._history) - 1
        # create QImage thumbnail
        w = self.doc.width()
        h = self.doc.height()
        qimg = QImage(raw, w, h, w*4, QImage.Format_ARGB32)

        if qimg.isNull():
            # fallback: don't add preview
            return

        thumb = qimg.scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pix = QPixmap.fromImage(thumb)
        # create an item with no visible text; show label as tooltip only
        item = QListWidgetItem()
        item.setIcon(QIcon(pix))
        tooltip = label if label else f"Snapshot {len(self._history)}"
        item.setToolTip(tooltip)
        # store raw bytes index in item data for quick restore
        item.setData(Qt.UserRole, len(self._history)-1)
        self.preview_list.addItem(item)
        try:
            self.preview_list.setCurrentRow(self._current_index)
        except Exception:
            pass

    def _on_preview_selected(self, item):
        idx = item.data(Qt.UserRole)
        if idx is None:
            self.append_chat('Selected preview has no data')
            return
        try:
            raw = self._history[int(idx)]
            # update current index and apply
            self._current_index = int(idx)
            self._apply_raw(raw)
            self.append_chat('Applied selected preview snapshot.')
        except Exception as e:
            self.append_chat('Failed to apply preview: ' + str(e))

    def run_gemini_processing(self):
        self.ok_button.setEnabled(False)
        try:
            api_key = self.get_api_key()
            prompt = self.get_prompt()

            if api_key:
                self.app.writeSetting("geminiPlugin", "apiKey", api_key)

            if not api_key:
                QMessageBox.warning(self, "Gemini", "API key is required.")
                return
            if not prompt:
                QMessageBox.warning(self, "Gemini", "Prompt cannot be empty.")
                return

            # Get node pixel raw bytes (for the full layer area)
            w = self.doc.width()
            h = self.doc.height()
            self.append_chat("Exporting active layer pixels...")
            raw = self.node.pixelData(0, 0, w, h)
            # push current snapshot onto history before sending to Gemini
            try:
                cur_raw = bytes(raw) if isinstance(raw, QByteArray) else raw
                # Keep Original snapshot appended to the end so new prompts
                # don't overwrite previews if the user has selected an earlier item.
                self._add_snapshot(cur_raw, label="Original", append_to_end=True)
            except Exception:
                pass
            # Convert raw RGBA bytes -> PNG bytes
            png_in = raw_rgba_to_png_bytes(raw, w, h)

            self.append_chat("Contacting Gemini...")
            out_png = call_gemini(api_key, prompt, png_in)
            self.append_chat("Received image from Gemini — applying to active layer...")

            # --- Alpha Channel Debugging ---
            self.append_chat("Checking received image for transparency...")
            qimg_test = QImage.fromData(QByteArray(out_png))
            if qimg_test.hasAlphaChannel():
                self.append_chat("--> SUCCESS: Image from API has an alpha channel.")
            else:
                self.append_chat("--> FAILURE: Image from API does NOT have an alpha channel.")
            # --------------------------------

            # Convert returned PNG bytes into raw RGBA bytes sized to document
            raw_out = png_bytes_to_raw_rgba(out_png, w, h)
            expected = w * h * 4
            self.append_chat(f"Prepared output bytes: {len(raw_out)} (expected {expected})")
            if len(raw_out) != expected:
                raise RuntimeError(f"Converted pixel buffer has wrong size: {len(raw_out)} != {expected}")
            # add the returned image as a new snapshot and preview
            try:
                # Always append Gemini results to the end of the preview stack
                self._add_snapshot(raw_out, append_to_end=True)
                # apply the raw_out
                self._apply_raw(raw_out)
                self.append_chat("Done. (snapshot added to preview queue)")
            except Exception as e:
                raise
        except Exception as e:
            self.append_chat("Error: " + str(e))
            QMessageBox.critical(self, "Gemini Error", str(e))
        finally:
            self.ok_button.setEnabled(True)

# ----- Utils: convert raw pixels -> PNG bytes ----- 
def raw_rgba_to_png_bytes(raw_bytes, w, h):
    if isinstance(raw_bytes, QByteArray):
        raw = bytes(raw_bytes)
    else:
        raw = raw_bytes
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw
    bytes_per_line = w * 4
    qimg = QImage(raw, w, h, bytes_per_line, QImage.Format_ARGB32)
    if qimg.isNull():
        qimg = QImage(raw, w, h, bytes_per_line, QImage.Format_RGBA8888)
    if qimg.isNull():
        raise RuntimeError("Could not create QImage from node.pixelData — unsupported format")
    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
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
        text_resp = None
        for p in parts:
            if "text" in p and p["text"]:
                text_resp = p["text"]; break
        raise RuntimeError("Gemini returned no image. " + (text_resp or ""))
    return base64.b64decode(img_b64)

def png_bytes_to_raw_rgba(png_bytes, w, h):
    qimg = QImage.fromData(QByteArray(png_bytes), "PNG") if hasattr(QImage, 'fromData') else QImage()
    if qimg.isNull():
        qimg = QImage()
        if not qimg.loadFromData(png_bytes, "PNG"):
            raise RuntimeError("Failed to load PNG image data from Gemini response")
    if qimg.width() != w or qimg.height() != h:
        qimg = qimg.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    # Convert to ARGB32 (BGRA) for consistency with Krita
    if qimg.format() != QImage.Format_ARGB32:
        qimg = qimg.convertToFormat(QImage.Format_ARGB32)
    byte_count = qimg.byteCount()
    try:
        ptr = qimg.bits()
        raw = ptr.asstring(byte_count)
    except Exception:
        buf = QBuffer()
        buf.open(QIODevice.ReadWrite)
        if not qimg.save(buf, "RAW"):
            raise RuntimeError("Unable to extract raw pixel bytes from QImage")
        raw = bytes(buf.data())
        buf.close()
    expected = w * h * 4
    if len(raw) < expected:
        raise RuntimeError(f"Not enough pixel data after conversion: got {len(raw)}, expected {expected}")
    if len(raw) > expected:
        raw = raw[:expected]
    return raw

# ----- Compatibility undo/transaction context ----- 
class _UndoContext:
    def __init__(self, doc, name):
        self.doc = doc
        self.name = name
        self._mode = None
        self._obj = None

    def __enter__(self):
        d = self.doc
        try:
            if hasattr(d, 'createCommand'):
                self._obj = d.createCommand(self.name)
                self._mode = 'createCommand'
                return self
        except Exception:
            pass
        try:
            if hasattr(d, 'openTransaction'):
                self._obj = d.openTransaction(self.name)
                self._mode = 'openTransaction'
                return self
        except Exception:
            pass
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
        self._mode = None
        return self

    def __exit__(self, exc_type, exc, tb):
        d = self.doc
        try:
            if self._mode == 'createCommand' and self._obj is not None:
                if exc_type is None:
                    try:
                        self._obj.commit()
                    except Exception:
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
                for end_name in ('endCommand', 'finishCommand', 'commitCommand', 'endTransaction'):
                    try:
                        if hasattr(d, end_name):
                            getattr(d, end_name)()
                            break
                    except Exception:
                        continue
        except Exception:
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

        saved_api_key = app.readSetting("geminiPlugin", "apiKey", "")
        dlg = GeminiDialog(doc, node, saved_api_key)
        dlg.exec_()

# Register the extension
app = Krita.instance()
extension = GeminiExtension(app)
app.addExtension(extension)
