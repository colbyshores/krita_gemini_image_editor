# gemini_krita.py
# Krita plugin: Gemini Image Editor (active-layer edit, undoable)
# Requirements: Krita (pykrita), PyQt5
# Place in Krita's pykrita/pykrita folder and restart Krita.

from krita import Krita, Extension
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QMessageBox, QListWidget, QListWidgetItem, QHBoxLayout, QSpinBox, QProgressDialog, QApplication, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPen
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice, Qt, QSize, QThread, pyqtSignal
import base64, json, urllib.request, urllib.error, hashlib

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"


class FrameGenWorker(QThread):
    """Background worker that generates frames by calling Gemini sequentially.

    Emits:
      progress(int, int) -> (index, total)
      finished(list) -> list of PNG bytes
      error(str) -> error message
    """
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

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
        self._stopped = True

    def run(self):
        try:
            results = []
            current_png = self.png_in
            for i in range(self.frames_count):
                if self._stopped:
                    break
                # Nudge the prompt and request that Gemini use the previous frame as the base.
                frame_prompt = (
                    self.prompt_base
                    + f" (animation frame {i+1} of {self.frames_count})"
                    + " -- continue the animation from the provided image, evolving it slightly to the next frame"
                )
                try:
                    guide_png = None
                    if self.use_guides:
                        # Compute a simple motion path from center -> offset and generate guide image
                        src = (self.w // 2, self.h // 2)
                        # Destination is a small offset to the right and slightly up
                        dst = (int(self.w * 0.2 + src[0]), int(src[1] - self.h * 0.1))
                        # For multiple guide points, create intermediate targets along the path
                        positions = []
                        for g in range(self.guide_count):
                            t = float(g) / max(1, self.guide_count - 1)
                            x = int(src[0] + (dst[0] - src[0]) * t)
                            y = int(src[1] + (dst[1] - src[1]) * t)
                            positions.append((x, y))
                        # Highlight the target for this frame (interpolated along path)
                        tframe = float(i + 1) / max(1, self.frames_count)
                        hx = int(src[0] + (dst[0] - src[0]) * tframe)
                        hy = int(src[1] + (dst[1] - src[1]) * tframe)
                        guide_png = create_motion_guide_png_bytes(self.w, self.h, positions, (hx, hy))

                    out = call_gemini(self.api_key, frame_prompt, current_png, guide_png)
                except Exception as e:
                    self.error.emit(str(e))
                    return
                # store and use this output as the base for the next iteration
                results.append(out)
                current_png = out
                self.progress.emit(i + 1, self.frames_count)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


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

        # Frames control for animation (1 = single image)
        self.frames_label = QLabel("Frames:")
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 60)
        self.frames_spin.setValue(1)
        # Frames-per-second control (for timeline settings)
        self.fps_label = QLabel("FPS:")
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(12)

        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)

        self.ok_button = QPushButton("Generate!")
        self.cancel_button = QPushButton("Close")
        # undo/redo buttons removed; preview queue handles rollbacks
        self.ok_button.clicked.connect(self.run_gemini_processing)
        self.cancel_button.clicked.connect(self.reject)

    # (preview queue removed) store last snapshot only
        self._last_snapshot = None

        # Left column controls
        left_v.addWidget(self.api_label)
        left_v.addWidget(self.api_entry)
        left_v.addWidget(self.prompt_label)
        left_v.addWidget(self.prompt_entry)
        left_v.addWidget(self.frames_label)
        left_v.addWidget(self.frames_spin)

        # Motion guide controls
        self.guides_checkbox = QCheckBox("Use motion guides")
        self.guide_count_label = QLabel("Guide points:")
        self.guide_count_spin = QSpinBox()
        self.guide_count_spin.setRange(1, 8)
        self.guide_count_spin.setValue(2)
        left_v.addWidget(self.guides_checkbox)
        left_v.addWidget(self.guide_count_label)
        left_v.addWidget(self.guide_count_spin)

        self.interpolate_checkbox = QCheckBox("Start animation from current image")
        self.interpolate_checkbox.setChecked(True)
        left_v.addWidget(self.interpolate_checkbox)

        left_v.addWidget(self.fps_label)
        left_v.addWidget(self.fps_spin)
        left_v.addWidget(QLabel("Chat / Status:"))
        left_v.addWidget(self.chat_view)
        left_v.addWidget(self.ok_button)
        left_v.addWidget(self.cancel_button)

        # Compose layouts
        main_h.addLayout(left_v)

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

    # _add_snapshot removed; plugin now stores only last snapshot in memory

    # Preview functionality removed; keep simple snapshot storage instead.

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
                self._last_snapshot = cur_raw
            except Exception:
                pass
            # Convert raw RGBA bytes -> PNG bytes
            png_in = raw_rgba_to_png_bytes(raw, w, h)

            frames_count = getattr(self, 'frames_spin', None) and int(self.frames_spin.value()) or 1
            interpolate = getattr(self, 'interpolate_checkbox', None) and bool(self.interpolate_checkbox.isChecked()) or False

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
                    self.ok_button.setEnabled(True)
                    return

                progress = QProgressDialog("Generating frames...", "Cancel", 0, frames_to_generate, self)
                progress.setWindowTitle("Generating Animation")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)

                # Worker will emit progress and finished signals
                use_guides = getattr(self, 'guides_checkbox', None) and bool(self.guides_checkbox.isChecked()) or False
                guide_count = getattr(self, 'guide_count_spin', None) and int(self.guide_count_spin.value()) or 2
                worker = FrameGenWorker(api_key, prompt, png_in, frames_to_generate, w, h, use_guides=use_guides, guide_count=guide_count, parent=self)

                def _on_progress(idx, total):
                    progress.setValue(idx)
                    self.append_chat(f"Contacting Gemini for frame {idx}/{total}...")

                def _on_finished(results):
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
                    progress.cancel()
                    self.append_chat(f"Frame generation error: {msg}")

                worker.progress.connect(_on_progress)
                worker.finished.connect(_on_finished)
                worker.error.connect(_on_error)

                def _on_cancel():
                    try:
                        worker.stop()
                    except Exception:
                        pass

                progress.canceled.connect(_on_cancel)
                worker.start()
                # keep worker referenced on the dialog so it doesn't get GC'd
                self._frame_worker = worker
                return

            use_guides = getattr(self, 'guides_checkbox', None) and bool(self.guides_checkbox.isChecked()) or False
            guide_count = getattr(self, 'guide_count_spin', None) and int(self.guide_count_spin.value()) or 2
            guide_png = None
            if use_guides:
                # simple guide for single frame: one destination offset
                src = (w // 2, h // 2)
                dst = (int(w * 0.2 + src[0]), int(src[1] - h * 0.1))
                positions = [src, dst]
                guide_png = create_motion_guide_png_bytes(w, h, positions, dst)
            self.append_chat("Contacting Gemini...")
            out_png = call_gemini(api_key, prompt, png_in, guide_png)
            self.append_chat("Received image from Gemini  applying to active layer...")

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
                # store last snapshot and apply
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

    def _create_animation_from_frames(self, raw_frames, label=None):
        """Create a group and one paint-layer per raw frame, then try to convert to timeline frames.

        Best-effort conversion: create layers under a group and attempt to trigger a conversion action.
        """
        doc = self.doc
        w = doc.width()
        h = doc.height()
        root = None
        try:
            root = doc.rootNode()
        except Exception:
            try:
                root = doc.topLevelNode()
            except Exception:
                root = None

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

        # Try to set fps on the document if available
        try:
            fps = getattr(self, 'fps_spin', None) and int(self.fps_spin.value()) or None
            if fps is not None and hasattr(doc, 'setAnimationFPS'):
                try:
                    doc.setAnimationFPS(fps)
                except Exception:
                    pass
        except Exception:
            pass

        converted = False
        try:
            app = Krita.instance()
            window = None
            try:
                window = app.activeWindow()
            except Exception:
                window = None

            # Use helper to try known action names and trigger the first available
            try:
                converted = find_and_trigger_convert_action(app, window, dialog=self)
            except Exception:
                converted = False

            # Fallback: try a convenience method on the group node itself
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
        # Try pinning created layers to timeline (Krita exposes pinning via layer properties in some builds)
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


def create_motion_guide_png_bytes(w, h, points, highlight_point=None):
    """Create a small transparent PNG with motion guide lines and points.

    - points: list of (x,y) tuples representing path anchors
    - highlight_point: (x,y) tuple to mark the current destination
    Returns PNG bytes.
    """
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    try:
        pen = QPen(QColor(255, 128, 0, 200))
        pen.setWidth(3)
        painter.setPen(pen)
        # Draw path lines
        if len(points) >= 2:
            for a, b in zip(points[:-1], points[1:]):
                painter.drawLine(a[0], a[1], b[0], b[1])
        # Draw small circles for anchor points
        for p in points:
            painter.setBrush(QColor(255, 200, 0, 180))
            painter.drawEllipse(p[0]-6, p[1]-6, 12, 12)
        # Highlight destination
        if highlight_point:
            hp = highlight_point
            pen2 = QPen(QColor(0, 200, 255, 220))
            pen2.setWidth(4)
            painter.setPen(pen2)
            painter.setBrush(QColor(0, 200, 255, 120))
            painter.drawEllipse(hp[0]-10, hp[1]-10, 20, 20)
    finally:
        painter.end()
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

# ----- Gemini call ----- 
def call_gemini(api_key, prompt, png_bytes, guide_png=None):
    # Backwards-compatible signature: callers may pass a 4th arg guide_png
    guide_png = None
    try:
        # If caller passed 4 args, capture guide_png from locals (FrameGenWorker does this)
        import inspect
        frame = inspect.currentframe().f_back
        args = frame.f_locals
        if 'guide_png' in args:
            guide_png = args.get('guide_png')
    except Exception:
        guide_png = None
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
    # If a guide image was passed, include it as a second inline_data part.
    if guide_png:
        try:
            body['contents'][0]['parts'].append({
                'text': 'motion_guide',
            })
            body['contents'][0]['parts'].append({
                'inline_data': {
                    'mime_type': 'image/png',
                    'data': base64.b64encode(guide_png).decode('utf-8')
                }
            })
        except Exception:
            pass
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


def find_and_trigger_convert_action(app, window, dialog=None):
    """Try a set of likely action names and trigger the first one that exists.

    Returns True if an action was found and triggered, False otherwise. If
    `dialog` is provided and has `append_chat`, the function will write a
    short status message so the user can see which action (if any) was used.
    """
    action_names = (
        'convertLayersToFrames',
        'convert_layers_to_frames',
        'animation.convert_layers_to_frames',
        'convertGroupToFrames',
        'convert_group_to_frames',
        'LayersToFrames',
        'layer_convert_to_frames',
    )

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
                    # If trigger failed, continue trying other names
                    continue
        except Exception:
            continue

    if dialog is not None and hasattr(dialog, 'append_chat'):
        dialog.append_chat("Automatic action trigger not found among known candidates.")
    return False

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
