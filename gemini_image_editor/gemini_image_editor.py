# gemini_krita_tabbed.py
# Krita plugin: Gemini Image Editor (Tabbed: Stills + Animation)
# Requirements: Krita (pykrita), PyQt5
# Place in Krita's `pykrita` folder and restart Krita.

# Krita and PyQt5 imports for plugin functionality and UI
from krita import Krita, Extension
from PyQt5.QtWidgets import (
    # UI Widgets
    QDialog, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox,
    QListWidget, QListWidgetItem, QHBoxLayout, QSpinBox, QProgressDialog, QApplication,
    QCheckBox, QTabWidget, QWidget
)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPen
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice, Qt, QSize, QThread, pyqtSignal
import base64, json, urllib.request, urllib.error, hashlib
from typing import List, Tuple

# The specific Gemini API endpoint for image generation.
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"


# =======================
# Shared worker & utils
# =======================

class FrameGenWorker(QThread):
    """A background worker thread to generate animation frames by calling the Gemini API.

    This prevents the Krita UI from freezing during long-running network requests.
      progress(int, int) -> (index, total)
      finished(list) -> list of PNG bytes
      error(str) -> error message
    """
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    # Store all necessary parameters for the generation task.
    def __init__(self, api_key, prompt_base, png_in, frames_count, w, h, use_guides=False, guide_count=2, parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.prompt_base = prompt_base
        self.png_in = png_in
        self.frames_count = int(frames_count)
        self.w = int(w)
        self.h = int(h)
        self.use_guides = bool(use_guides)
        self.guide_count = int(guide_count)
        self._stopped = False

    def stop(self):
        """Flags the worker to stop processing before the next frame."""
        self._stopped = True

    def run(self):
        """The main execution method of the thread."""
        try:
            results = []
            # The first frame is the input image. Subsequent frames use the previously generated one.
            current_png = self.png_in
            for i in range(self.frames_count):
                if self._stopped:
                    break
                # Append frame-specific information to the base prompt.
                frame_prompt = (
                    self.prompt_base
                    + f" (animation frame {i+1} of {self.frames_count})"
                    + " -- continue the animation from the provided image, evolving it slightly to the next frame"
                )
                try:
                    guide_png = None
                    if self.use_guides:
                        # If motion guides are enabled, compute a simple path and create a guide image.
                        src = (self.w // 2, self.h // 2)
                        dst = (int(self.w * 0.2 + src[0]), int(src[1] - self.h * 0.1))
                        positions = []
                        for g in range(self.guide_count):
                            t = float(g) / max(1, self.guide_count - 1)
                            x = int(src[0] + (dst[0] - src[0]) * t)
                            y = int(src[1] + (dst[1] - src[1]) * t)
                            positions.append((x, y))
                        tframe = float(i + 1) / max(1, self.frames_count)
                        hx = int(src[0] + (dst[0] - src[0]) * tframe)
                        hy = int(src[1] + (dst[1] - src[1]) * tframe)
                        guide_png = create_motion_guide_png_bytes(self.w, self.h, positions, (hx, hy))

                    # Call the Gemini API to generate the next frame.
                    out = call_gemini(self.api_key, frame_prompt, current_png, guide_png=guide_png)
                except Exception as e:
                    self.error.emit(str(e))
                    return
                # Add the new frame to our results and set it as the input for the next iteration.
                results.append(out)
                current_png = out
                # Emit progress to update the UI.
                self.progress.emit(i + 1, self.frames_count)
            # Once all frames are generated, emit the finished signal with the results.
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ----- Utils: convert raw pixels <-> PNG bytes -----

def raw_rgba_to_png_bytes(raw_bytes: bytes, w: int, h: int) -> bytes:
    """Converts raw RGBA pixel data from Krita into standard PNG image bytes.

    Args:
        raw_bytes: The raw pixel data (usually from `node.pixelData`).
        w: The width of the image.
        h: The height of the image.
    """
    if isinstance(raw_bytes, QByteArray):
        raw = bytes(raw_bytes)
    else:
        raw = raw_bytes
    if len(raw) >= 8 and raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw
    bytes_per_line = w * 4
    # Create a QImage from the raw data. Krita often uses ARGB, but we try RGBA as a fallback.
    qimg = QImage(raw, w, h, bytes_per_line, QImage.Format_ARGB32)
    if qimg.isNull():
        qimg = QImage(raw, w, h, bytes_per_line, QImage.Format_RGBA8888)
    if qimg.isNull():
        raise RuntimeError("Could not create QImage from node.pixelData — unsupported format")

    # Save the QImage to an in-memory buffer in PNG format.
    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
    ok = qimg.save(buffer, "PNG")
    buffer.seek(0)
    if not ok:
        raise RuntimeError("Failed to save QImage to PNG bytes")

    # Extract the bytes from the buffer.
    png_bytes = buffer.data().data() if hasattr(buffer.data(), "data") else bytes(buffer.data())
    buffer.close()
    return png_bytes


def create_motion_guide_png_bytes(w: int, h: int, points: List[Tuple[int, int]], highlight_point: Tuple[int, int] = None) -> bytes:
    """Creates a transparent PNG image with visual motion guides.

    This generates an image with lines and circles that can be passed to the AI
    to give it hints about desired motion or object paths.

    Args:
        w: Width of the guide image.
        h: Height of the guide image.
        points: A list of (x, y) tuples representing points on a path.
        highlight_point: An optional (x, y) tuple for the current frame's position, drawn larger.

    Returns:
        The motion guide image as PNG bytes.
    """
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    try:
        pen = QPen(QColor(255, 128, 0, 200))
        pen.setWidth(3)
        painter.setPen(pen)
        # Draw lines connecting the points to form a path.
        if len(points) >= 2:
            for a, b in zip(points[:-1], points[1:]):
                painter.drawLine(a[0], a[1], b[0], b[1])
        for p in points:
            painter.setBrush(QColor(255, 200, 0, 180))
            # Draw a small circle at each point.
            painter.drawEllipse(p[0]-6, p[1]-6, 12, 12)
        if highlight_point:
            hp = highlight_point
            pen2 = QPen(QColor(0, 200, 255, 220))
            pen2.setWidth(4)
            painter.setPen(pen2)
            painter.setBrush(QColor(0, 200, 255, 120))
            # Draw a larger, highlighted circle for the current frame's position.
            painter.drawEllipse(hp[0]-10, hp[1]-10, 20, 20)
    finally:
        painter.end()

    # Save the painted image to a buffer as a PNG.
    buf = QBuffer()
    buf.open(QIODevice.ReadWrite)
    ok = img.save(buf, 'PNG')
    buf.seek(0)
    if not ok:
        buf.close()
        raise RuntimeError('Failed to create guide PNG')
    out = buf.data().data() if hasattr(buf.data(), 'data') else bytes(buf.data())
    buf.close()
    return out


def call_gemini(api_key: str, prompt: str, png_bytes: bytes, guide_png: bytes = None) -> bytes:
    """Constructs the request and calls the Google Gemini API.

    Args:
        api_key: The user's Google Gemini API key.
        prompt: The text prompt describing the desired edit.
        png_bytes: The source image as PNG bytes.
        guide_png: Optional motion guide image as PNG bytes.
    """
    # Build the JSON payload for the API request.
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
    # If a motion guide is provided, add it to the request parts.
    if guide_png:
        body['contents'][0]['parts'].append({ 'text': 'motion_guide' })
        body['contents'][0]['parts'].append({
            'inline_data': {
                'mime_type': 'image/png',
                'data': base64.b64encode(guide_png).decode('utf-8')
            }
        })

    # Prepare and send the HTTP POST request.
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

    # Parse the response to find the returned image data.
    img_b64 = None
    try:
        parts = resp_data["candidates"][0]["content"]["parts"]
        for p in parts:
            if "inline_data" in p and p["inline_data"].get("data"):
                img_b64 = p["inline_data"]["data"]
                break
            # The API sometimes uses camelCase 'inlineData'.
            if "inlineData" in p and p["inlineData"].get("data"):
                img_b64 = p["inlineData"]["data"]
                break
    except Exception:
        raise RuntimeError("Unexpected Gemini response format")
    if not img_b64:
        text_resp = None
        # If no image is returned, try to find a text response to show the user.
        for p in parts:
            if "text" in p and p["text"]:
                text_resp = p["text"]; break
        raise RuntimeError("Gemini returned no image. " + (text_resp or ""))

    # Decode the base64 image data and return it.
    return base64.b64decode(img_b64)


def png_bytes_to_raw_rgba(png_bytes: bytes, w: int, h: int) -> bytes:
    """Converts PNG image bytes (from Gemini) back to raw RGBA pixel data for Krita.

    Args:
        png_bytes: The PNG image data.
        w: The target width for the raw data (document width).
        h: The target height for the raw data (document height).
    """
    # Load the PNG data into a QImage.
    qimg = QImage.fromData(QByteArray(png_bytes), "PNG") if hasattr(QImage, 'fromData') else QImage()
    if qimg.isNull():
        qimg = QImage()
        if not qimg.loadFromData(png_bytes, "PNG"):
            raise RuntimeError("Failed to load PNG image data from Gemini response")

    # Ensure the image dimensions match the Krita document.
    if qimg.width() != w or qimg.height() != h:
        qimg = qimg.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    # Ensure the image is in a format Krita can easily use (ARGB32).
    if qimg.format() != QImage.Format_ARGB32:
        qimg = qimg.convertToFormat(QImage.Format_ARGB32)

    # Extract the raw bytes from the QImage.
    byte_count = qimg.byteCount()
    try:
        # The 'bits().asstring()' method is fast but can be fragile across Qt versions.
        ptr = qimg.bits()
        raw = ptr.asstring(byte_count)
    except Exception:
        # A more robust fallback is to save to a buffer in a raw format.
        buf = QBuffer()
        buf.open(QIODevice.ReadWrite)
        if not qimg.save(buf, "RAW"):
            raise RuntimeError("Unable to extract raw pixel bytes from QImage")
        raw = bytes(buf.data())
        buf.close()
    expected = w * h * 4

    # Final check to ensure the byte buffer is the correct size.
    if len(raw) < expected:
        raise RuntimeError(f"Not enough pixel data after conversion: got {len(raw)}, expected {expected}")
    if len(raw) > expected:
        raw = raw[:expected]
    return raw


def find_and_trigger_convert_action(app, window, dialog=None):
    """Tries to find and trigger Krita's "Convert Layers to Frames" action.

    Krita's internal action names can vary, so this function tries a list of
    known candidates to provide better compatibility.
    """
    # A list of possible internal names for the action.
    action_names = (
        'convertLayersToFrames',
        'convert_layers_to_frames',
        'animation.convert_layers_to_frames',
        'convertGroupToFrames',
        'convert_group_to_frames',
        'LayersToFrames',
        'layer_convert_to_frames',
    )

    # Iterate through the names and try to find and trigger the action.
    for act_name in action_names:
        try:
            act = None
            if window is not None and hasattr(window, 'action'):
                try:
                    act = window.action(act_name)
                except Exception:
                    act = None
            if act is None and hasattr(app, 'action'):
                try:
                    act = app.action(act_name)
                except Exception:
                    act = None

            if act is not None:
                try:
                    act.trigger()
                    if dialog is not None and hasattr(dialog, 'append_chat'):
                        dialog.append_chat(f"Triggered Krita action '{act_name}' to convert layers -> frames.")
                    return True
                except Exception:
                    continue
        except Exception:
            continue

    if dialog is not None and hasattr(dialog, 'append_chat'):
        dialog.append_chat("Automatic action trigger not found among known candidates.")
    return False


# ----- Compatibility undo/transaction context -----
class _UndoContext:
    """A context manager to wrap an operation in a single Krita undo step.

    This class handles API variations between different Krita versions for creating
    and managing undo transactions, making the plugin more robust.
    """
    def __init__(self, doc, name):
        """Initializes the context with the document and a name for the undo step."""
        self.doc = doc
        self.name = name
        self._mode = None
        self._obj = None

    def __enter__(self):
        d = self.doc
        # Try different methods to start a transaction, from newest to oldest APIs.
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
        # Based on which 'start' method succeeded, call the corresponding 'end' method.
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
                        # If an exception occurred, we rollback/abort the transaction.
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


# =======================
# Stills (Preview) Tab
# =======================
class StillsPanel(QWidget):
    """The UI and logic for the 'Stills (Preview)' tab."""
    def __init__(self, doc, node, saved_key="", parent=None):
        super().__init__(parent)
        self.doc = doc
        self.node = node
        self.app = Krita.instance()

        # --- UI Setup ---
        main_h = QHBoxLayout(self)
        left_v = QVBoxLayout()
        right_v = QVBoxLayout()

        # Left side: Controls (API key, prompt, status, generate button)
        self.api_label = QLabel("API Key:")
        self.api_entry = QLineEdit()
        self.api_entry.setEchoMode(QLineEdit.Password)
        self.api_entry.setText(saved_key)

        self.prompt_label = QLabel("Prompt:")
        self.prompt_entry = QLineEdit()

        self.chat_view = QTextEdit(); self.chat_view.setReadOnly(True)

        self.ok_button = QPushButton("Generate Still")
        self.ok_button.clicked.connect(self.run_gemini_processing)

        left_v.addWidget(self.api_label)
        left_v.addWidget(self.api_entry)
        left_v.addWidget(self.prompt_label)
        left_v.addWidget(self.prompt_entry)
        left_v.addWidget(QLabel("Chat / Status:"))
        left_v.addWidget(self.chat_view)
        left_v.addWidget(self.ok_button)

        # Right side: A list widget to show a queue of generated image previews.
        self.preview_list = QListWidget()
        self.preview_list.setIconSize(QSize(160, 90))
        self.preview_list.setFixedWidth(180)
        self.preview_list.itemClicked.connect(self._on_preview_selected)
        right_v.addWidget(QLabel("Preview Queue"))
        right_v.addWidget(self.preview_list)

        main_h.addLayout(left_v)
        main_h.addLayout(right_v)

        # --- History Data Structures ---
        # Stores raw pixel data for each generated snapshot.
        self._history = []  # list[bytes]
        # Points to the currently displayed snapshot in the history.
        self._current_index = -1
        # A map of image hashes to their index to prevent duplicates.
        self._hash_map = {}

        self.append_chat("Ready. This tab generates stills and manages a preview queue.")

    def append_chat(self, text):
        """Appends a message to the status/chat text view."""
        self.chat_view.append(text)
        self.chat_view.repaint()

    def _apply_raw(self, raw_bytes):
        """Applies raw pixel data to the active Krita layer."""
        w = self.doc.width(); h = self.doc.height()
        expected = w * h * 4
        if len(raw_bytes) != expected:
            raise RuntimeError(f"Snapshot buffer size mismatch: {len(raw_bytes)} != {expected}")
        # Wrap the operation in an undo context so it can be undone with Ctrl+Z.
        with _UndoContext(self.doc, "Gemini Image Edit (stills)"):
            qba = QByteArray(raw_bytes)
            self.node.setPixelData(qba, 0, 0, w, h)
            self.doc.refreshProjection()

    def _add_snapshot(self, raw_bytes, label=None, append_to_end=False):
        """Adds an image snapshot to the history and the preview list UI.

        This allows the user to quickly switch between different generated versions.
        """
        raw = bytes(raw_bytes) if isinstance(raw_bytes, QByteArray) else raw_bytes

        if not append_to_end and self._current_index < len(self._history) - 1:
            keep = self._current_index + 1
            while self.preview_list.count() > keep:
                itm = self.preview_list.takeItem(self.preview_list.count() - 1)
                try: del itm
                except Exception: pass
            self._history = self._history[:keep]
            self._hash_map = {hashlib.sha256(hraw).hexdigest(): i for i, hraw in enumerate(self._history)}
            while self.preview_list.count() > len(self._history):
                itm = self.preview_list.takeItem(self.preview_list.count() - 1)
                try: del itm
                except Exception: pass

        # Check if this exact image already exists in the history to avoid duplicates.
        new_hash = hashlib.sha256(raw).hexdigest()
        if new_hash in self._hash_map:
            self._current_index = self._hash_map[new_hash]
            try: self.preview_list.setCurrentRow(self._current_index)
            except Exception: pass
            return

        # Add the new snapshot to history.
        self._history.append(raw)
        self._hash_map[new_hash] = len(self._history) - 1
        self._current_index = len(self._history) - 1

        w = self.doc.width(); h = self.doc.height()
        # Create a thumbnail for the preview list.
        qimg = QImage(raw, w, h, w*4, QImage.Format_ARGB32)
        if qimg.isNull():
            return
        thumb = qimg.scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pix = QPixmap.fromImage(thumb)
        item = QListWidgetItem(); item.setIcon(QIcon(pix))
        tooltip = label if label else f"Snapshot {len(self._history)}"
        item.setToolTip(tooltip)
        item.setData(Qt.UserRole, len(self._history)-1)
        self.preview_list.addItem(item)
        try: self.preview_list.setCurrentRow(self._current_index)
        except Exception: pass

    def _on_preview_selected(self, item):
        """Slot that is called when a user clicks on a preview thumbnail."""
        idx = item.data(Qt.UserRole)
        if idx is None:
            self.append_chat('Selected preview has no data')
            return
        try:
            raw = self._history[int(idx)]
            self._current_index = int(idx)
            self._apply_raw(raw)
            self.append_chat('Applied selected preview snapshot.')
        except Exception as e:
            self.append_chat('Failed to apply preview: ' + str(e))

    def run_gemini_processing(self):
        """The main function called when the 'Generate Still' button is clicked."""
        self.ok_button.setEnabled(False)
        try:
            api_key = self.api_entry.text().strip()
            prompt = self.prompt_entry.text().strip()

            # Validate inputs.
            if api_key:
                self.app.writeSetting("geminiPlugin", "apiKey", api_key)
            if not api_key:
                QMessageBox.warning(self, "Gemini", "API key is required.")
                return
            if not prompt:
                QMessageBox.warning(self, "Gemini", "Prompt cannot be empty.")
                return

            # Get the current layer's image data.
            w = self.doc.width(); h = self.doc.height()
            self.append_chat("Exporting active layer pixels...")
            raw = self.node.pixelData(0, 0, w, h)
            try:
                cur_raw = bytes(raw) if isinstance(raw, QByteArray) else raw
                self._add_snapshot(cur_raw, label="Original", append_to_end=True)
            except Exception:
                pass
            png_in = raw_rgba_to_png_bytes(raw, w, h)

            # Call the API and process the result.
            self.append_chat("Contacting Gemini...")
            out_png = call_gemini(api_key, prompt, png_in)
            self.append_chat("Received image from Gemini — applying to active layer...")

            # Check if the returned image has transparency.
            qimg_test = QImage.fromData(QByteArray(out_png))
            if qimg_test.hasAlphaChannel():
                self.append_chat("--> SUCCESS: Image from API has an alpha channel.")
            else:
                self.append_chat("--> WARNING: Image from API does NOT have an alpha channel.")

            raw_out = png_bytes_to_raw_rgba(out_png, w, h)
            expected = w * h * 4
            self.append_chat(f"Prepared output bytes: {len(raw_out)} (expected {expected})")
            if len(raw_out) != expected:
                raise RuntimeError(f"Converted pixel buffer has wrong size: {len(raw_out)} != {expected}")

            # Add the new image to the preview queue and apply it to the canvas.
            try:
                self._add_snapshot(raw_out, append_to_end=True)
                self._apply_raw(raw_out)
                self.append_chat("Done. (snapshot added to preview queue)")
            except Exception as e:
                raise
        except Exception as e:
            self.append_chat("Error: " + str(e))
            QMessageBox.critical(self, "Gemini Error", str(e))
        finally:
            self.ok_button.setEnabled(True)


# =======================
# Animation Tab
# =======================
class AnimationPanel(QWidget):
    """The UI and logic for the 'Animation' tab."""
    def __init__(self, doc, node, saved_key="", parent=None):
        super().__init__(parent)
        self.doc = doc
        self.node = node
        self.app = Krita.instance()
        self._last_snapshot = None

        main_h = QHBoxLayout(self)
        # --- UI Setup ---
        left_v = QVBoxLayout()

        self.api_label = QLabel("API Key:")
        self.api_entry = QLineEdit(); self.api_entry.setEchoMode(QLineEdit.Password); self.api_entry.setText(saved_key)

        self.prompt_label = QLabel("Prompt:")
        self.prompt_entry = QLineEdit()

        self.frames_label = QLabel("Frames:")
        self.frames_spin = QSpinBox(); self.frames_spin.setRange(1, 60); self.frames_spin.setValue(1)

        self.fps_label = QLabel("FPS:")
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 60); self.fps_spin.setValue(12)

        self.guides_checkbox = QCheckBox("Use motion guides")
        self.guide_count_label = QLabel("Guide points:")
        self.guide_count_spin = QSpinBox(); self.guide_count_spin.setRange(1, 8); self.guide_count_spin.setValue(2)

        self.interpolate_checkbox = QCheckBox("Start animation from current image")
        self.interpolate_checkbox.setChecked(True)

        self.chat_view = QTextEdit(); self.chat_view.setReadOnly(True)

        self.ok_button = QPushButton("Generate Animation")
        self.ok_button.clicked.connect(self.run_gemini_processing)

        left_v.addWidget(self.api_label)
        left_v.addWidget(self.api_entry)
        left_v.addWidget(self.prompt_label)
        left_v.addWidget(self.prompt_entry)
        left_v.addWidget(self.frames_label)
        left_v.addWidget(self.frames_spin)
        left_v.addWidget(self.fps_label)
        left_v.addWidget(self.fps_spin)
        left_v.addWidget(self.guides_checkbox)
        left_v.addWidget(self.guide_count_label)
        left_v.addWidget(self.guide_count_spin)
        left_v.addWidget(self.interpolate_checkbox)
        left_v.addWidget(QLabel("Chat / Status:"))
        left_v.addWidget(self.chat_view)
        left_v.addWidget(self.ok_button)

        main_h.addLayout(left_v)

        self.append_chat("Ready. This tab generates animations (layers -> timeline).")

        self._frame_worker = None

    def append_chat(self, text):
        """Appends a message to the status/chat text view."""
        self.chat_view.append(text)
        self.chat_view.repaint()

    def _apply_raw(self, raw_bytes):
        """Applies raw pixel data to the active Krita layer."""
        w = self.doc.width(); h = self.doc.height()
        expected = w * h * 4
        if len(raw_bytes) != expected:
            raise RuntimeError(f"Snapshot buffer size mismatch: {len(raw_bytes)} != {expected}")
        # Wrap in an undo context.
        with _UndoContext(self.doc, "Gemini Image Edit (animation)"):
            qba = QByteArray(raw_bytes)
            self.node.setPixelData(qba, 0, 0, w, h)
            self.doc.refreshProjection()

    def _create_animation_from_frames(self, raw_frames, label=None):
        """Takes a list of raw image frames and creates new layers in Krita for them.

        This function attempts to create a new group, add a paint layer for each
        frame, and then trigger Krita's action to convert these layers into an
        animation on the timeline.
        """
        doc = self.doc
        w = doc.width(); h = doc.height()

        # Find the root node of the layer stack.
        try:
            root = doc.rootNode()
        except Exception:
            try:
                root = doc.topLevelNode()
            except Exception:
                root = None

        # Create a new group layer to hold the animation frames.
        group_node = None
        group_name = (label[:40] + "...") if label and len(label) > 40 else (label or "Generated Animation")
        try:
            if hasattr(doc, 'createNode'):
                for typ in ('groupLayer', 'grouplayer', 'group'):
                    try:
                        try:
                            group_node = doc.createNode(group_name, typ, doc)
                        except TypeError:
                            group_node = doc.createNode(group_name, typ)
                        break
                    except Exception:
                        group_node = None
                if group_node is not None and root is not None and hasattr(root, 'addChildNode'):
                    try:
                        root.addChildNode(group_node, None)
                    except Exception:
                        pass
        except Exception:
            group_node = None

        # For each frame, create a new paint layer and set its pixel data.
        created_nodes = []
        for i, raw in enumerate(raw_frames):
            node_name = f"{group_name} - frame {i+1}"
            new_node = None
            try:
                if hasattr(doc, 'createNode'):
                    for typ in ('paintLayer', 'paintlayer', 'paint'):
                        try:
                            try:
                                new_node = doc.createNode(node_name, typ, doc)
                            except TypeError:
                                new_node = doc.createNode(node_name, typ)
                            break
                        except Exception:
                            new_node = None
                if new_node is None and hasattr(self.node, 'duplicate'):
                    try:
                        new_node = self.node.duplicate()
                    except Exception:
                        new_node = None

                if new_node is None:
                    self.append_chat(f"Could not create layer for frame {i+1}; skipping.")
                    continue

                parent = group_node if group_node is not None else root
                if parent is not None and hasattr(parent, 'addChildNode'):
                    try:
                        parent.addChildNode(new_node, None)
                    except Exception:
                        pass

                try:
                    qba = QByteArray(raw)
                    if hasattr(new_node, 'setPixelData'):
                        new_node.setPixelData(qba, 0, 0, w, h)
                    elif hasattr(new_node, 'setPixels'):
                        new_node.setPixels(qba)
                except Exception:
                    pass

                created_nodes.append(new_node)
            except Exception as e:
                self.append_chat(f"Failed to create/apply frame {i+1}: {e}")

        try:
            doc.refreshProjection()
        except Exception:
            pass

        # Set the document's FPS to the value specified in the UI.
        try:
            fps = int(self.fps_spin.value())
            if hasattr(doc, 'setAnimationFPS'):
                try: doc.setAnimationFPS(fps)
                except Exception: pass
        except Exception:
            pass

        # Try to automatically convert the newly created layers into timeline frames.
        converted = False
        try:
            app = Krita.instance()
            try:
                window = app.activeWindow()
            except Exception:
                window = None
            try:
                converted = find_and_trigger_convert_action(app, window, dialog=self)
            except Exception:
                converted = False
            # As a fallback, try a method directly on the group node if it exists.
            if not converted and group_node is not None:
                if hasattr(group_node, 'convertToFrames'):
                    try:
                        group_node.convertToFrames()
                        converted = True
                        self.append_chat("Used group_node.convertToFrames() as a fallback.")
                    except Exception:
                        pass
        except Exception:
            pass

        # Try to "pin" the layers to the timeline, which is sometimes necessary.
        try:
            for n in created_nodes:
                try:
                    if hasattr(n, 'setPinnedToTimeline'):
                        n.setPinnedToTimeline(True)
                    elif hasattr(n, 'pinToTimeline'):
                        n.pinToTimeline(True)
                except Exception:
                    continue
        except Exception:
            pass

        if not converted:
            self.append_chat("Note: automatic conversion to timeline frames not available; layers were created inside a group.")

    def run_gemini_processing(self):
        """The main function called when the 'Generate Animation' button is clicked."""
        self.ok_button.setEnabled(False)
        try:
            api_key = self.api_entry.text().strip()
            prompt = self.prompt_entry.text().strip()

            # Validate inputs.
            if api_key:
                self.app.writeSetting("geminiPlugin", "apiKey", api_key)
            if not api_key:
                QMessageBox.warning(self, "Gemini", "API key is required.")
                return
            if not prompt:
                QMessageBox.warning(self, "Gemini", "Prompt cannot be empty.")
                return

            # Get current layer's image data.
            w = self.doc.width(); h = self.doc.height()
            self.append_chat("Exporting active layer pixels...")
            raw = self.node.pixelData(0, 0, w, h)
            try:
                cur_raw = bytes(raw) if isinstance(raw, QByteArray) else raw
                self._last_snapshot = cur_raw
            except Exception:
                pass
            png_in = raw_rgba_to_png_bytes(raw, w, h)

            frames_count = int(self.frames_spin.value()) if self.frames_spin else 1
            interpolate = bool(self.interpolate_checkbox.isChecked()) if self.interpolate_checkbox else False

            # --- Multi-frame animation logic ---
            if frames_count > 1:
                self.append_chat(f"Generating {frames_count} animation frames from prompt...")

                initial_raw = None
                frames_to_generate = frames_count
                if interpolate:
                    initial_raw = bytes(raw)
                    frames_to_generate = frames_count - 1

                if frames_to_generate < 1:
                    if initial_raw:
                        self._create_animation_from_frames([initial_raw], label=prompt)
                        self.append_chat("Done. Single-frame animation created from current image.")
                    else:
                        self.append_chat("Not enough frames to generate.")
                    return

                # Show a progress dialog to the user.
                progress = QProgressDialog("Generating frames...", "Cancel", 0, frames_to_generate, self)
                progress.setWindowTitle("Generating Animation")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)

                use_guides = bool(self.guides_checkbox.isChecked()) if self.guides_checkbox else False
                guide_count = int(self.guide_count_spin.value()) if self.guide_count_spin else 2
                # Create and configure the background worker.
                worker = FrameGenWorker(api_key, prompt, png_in, frames_to_generate, w, h, use_guides=use_guides, guide_count=guide_count, parent=self)

                # --- Worker Signal Handlers ---
                def _on_progress(idx, total):
                    """Updates the progress bar."""
                    progress.setValue(idx)
                    self.append_chat(f"Contacting Gemini for frame {idx}/{total}...")

                def _on_finished(results):
                    """Called when the worker successfully generates all frames."""
                    progress.setValue(frames_to_generate)
                    if not results and not initial_raw:
                        self.append_chat("No frames were generated.")
                        return
                    try:
                        raw_frames = [png_bytes_to_raw_rgba(fp, w, h) for fp in results]
                        if initial_raw:
                            raw_frames.insert(0, initial_raw)
                        self._create_animation_from_frames(raw_frames, label=prompt)
                        try:
                            self._apply_raw(raw_frames[0])
                        except Exception:
                            pass
                        self.append_chat("Done. Animation frames added (group/layers).")
                    except Exception as e:
                        self.append_chat(f"Error processing frames: {e}")

                def _on_error(msg):
                    """Called if the worker encounters an error."""
                    progress.cancel()
                    self.append_chat(f"Frame generation error: {msg}")

                # Connect signals to the handler functions.
                worker.progress.connect(_on_progress)
                worker.finished.connect(_on_finished)
                worker.error.connect(_on_error)

                def _on_cancel():
                    """Stops the worker if the user clicks 'Cancel' on the progress dialog."""
                    try: worker.stop()
                    except Exception: pass

                progress.canceled.connect(_on_cancel)
                # Start the background worker.
                worker.start()
                self._frame_worker = worker  # keep reference
                return

            # --- Single-frame generation logic (if frames_count is 1) ---
            use_guides = bool(self.guides_checkbox.isChecked()) if self.guides_checkbox else False
            guide_count = int(self.guide_count_spin.value()) if self.guide_count_spin else 2
            guide_png = None
            if use_guides:
                src = (w // 2, h // 2)
                dst = (int(w * 0.2 + src[0]), int(src[1] - h * 0.1))
                positions = [src, dst]
                guide_png = create_motion_guide_png_bytes(w, h, positions, dst)

            self.append_chat("Contacting Gemini...")
            # Call the API directly since it's just one frame.
            out_png = call_gemini(api_key, prompt, png_in, guide_png=guide_png)
            self.append_chat("Received image from Gemini applying to active layer...")

            # Check for alpha channel and convert back to raw pixels.
            qimg_test = QImage.fromData(QByteArray(out_png))
            if qimg_test.hasAlphaChannel():
                self.append_chat("--> SUCCESS: Image from API has an alpha channel.")
            else:
                self.append_chat("--> WARNING: Image from API does NOT have an alpha channel.")

            raw_out = png_bytes_to_raw_rgba(out_png, w, h)
            expected = w * h * 4
            self.append_chat(f"Prepared output bytes: {len(raw_out)} (expected {expected})")
            if len(raw_out) != expected:
                raise RuntimeError(f"Converted pixel buffer has wrong size: {len(raw_out)} != {expected}")

            # Apply the result to the active layer.
            try:
                self._last_snapshot = raw_out
                self._apply_raw(raw_out)
                self.append_chat("Done.")
            except Exception as e:
                raise
        except Exception as e:
            self.append_chat("Error: " + str(e))
            QMessageBox.critical(self, "Gemini Error", str(e))
        finally:
            self.ok_button.setEnabled(True)


# =======================
# Main dialog with tabs
# =======================
class GeminiTabbedDialog(QDialog):
    """The main dialog window for the plugin, containing the Stills and Animation tabs."""
    def __init__(self, doc, node, saved_key="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gemini Image Editor")
        self.resize(900, 520)

        # Create the tab widget.
        tabs = QTabWidget()
        # Create instances of our two panels.
        self.stills_tab = StillsPanel(doc, node, saved_key, parent=self)
        self.anim_tab = AnimationPanel(doc, node, saved_key, parent=self)
        tabs.addTab(self.stills_tab, "Stills (Preview)")
        tabs.addTab(self.anim_tab, "Animation")

        # Close button for the whole dialog
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)

        # Set the main layout for the dialog.
        root = QVBoxLayout(self)
        root.addWidget(tabs)
        root.addWidget(close_btn)


# =======================
# Krita extension
# =======================
class GeminiExtension(Extension):
    """The main Krita Extension class that integrates the plugin into Krita."""
    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        """Called by Krita to set up the extension. Currently does nothing."""
        pass

    def createActions(self, window):
        """Called by Krita to create menu actions."""
        # Create an action that will appear in the "Tools > Scripts" menu.
        action = window.createAction("gemini_image_editor", "Gemini Image Editor...", "tools/scripts")
        # Connect the action's trigger to our `run` method.
        action.triggered.connect(lambda: self.run(window))

    def run(self, window):
        """This method is executed when the user clicks the menu item."""
        app = Krita.instance()
        doc = app.activeDocument()
        # Check if there is an open document.
        if not doc:
            QMessageBox.warning(None, "Gemini", "No active document.")
            return

        node = doc.activeNode()
        if not node:
            QMessageBox.warning(None, "Gemini", "No active layer selected.")
            return

        # Read the saved API key from Krita's settings to pre-fill the field.
        saved_api_key = app.readSetting("geminiPlugin", "apiKey", "")
        # Create and show the main dialog.
        dlg = GeminiTabbedDialog(doc, node, saved_api_key)
        dlg.exec_()


# This block is executed when Krita loads the plugin. It creates an instance
# of our extension and registers it with the Krita application.
app = Krita.instance()
extension = GeminiExtension(app)
app.addExtension(extension)
