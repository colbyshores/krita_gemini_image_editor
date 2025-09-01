# Krita Plugin Packaging Guide

This project provides instructions on how to properly package and distribute a Krita Python plugin.  
It has been tested using **Krita 5.0.2 on Pop!\_OS 22.04** (Ubuntu-based Linux distribution).

---

## 📦 Packaging a Krita Plugin

Krita Python plugins follow a specific structure to be recognized and loaded correctly by Krita.

### 1. Directory Structure

Your plugin be a zip file and have the following layout:

```

./
├── gemini_image_editor.desktop
├── gemini_image_editor/
    ├── **init**.py
    ├── gemini_image_editor.py
└── README.md
└── LICENSE.md

````

**Key files:**
- `gemini_image_editor.desktop` → Metadata file used by Krita to display the plugin under `Tools → Scripts → Gemini Image Editor`.
- `__init__.py` → Marks the folder as a Python module.
- `gemini_image_editor.py` → the main plugin logic.


---

### 3. Installing the Plugin

On **Pop!\_OS 22.04 (or Ubuntu-based systems)**, Krita plugins are installed under:

```
~/.local/share/krita/pykrita/
```

To install:

```bash
mkdir -p ~/.local/share/krita/pykrita
cp -r gemini_image_editor ~/.local/share/krita/pykrita/
```

Restart Krita, then:

1. Go to `Settings → Configure Krita → Python Plugins`.
2. Enable your plugin.
3. It will now appear under `Tools → Scripts → Gemini Image Editor`.

---

### 4. Distributing the Plugin

For distribution:

1. Zip your plugin directory:

   ```bash
   zip -r gemini_image_editor.zip my_plugin/
   ```
2. Share `gemini_image_editor.zip`.
   Users can unzip and place it inside their Krita `pykrita` folder.

---

## ✅ Tested Environment

* **Krita:** 5.0.2
* **OS:** Pop!\_OS 22.04 (Ubuntu-based Linux)

Functionality and packaging confirmed working in this setup. Behavior on other systems or newer versions of Krita may vary.

---

## 📖 References

* [Krita Python Scripting Documentation](https://docs.krita.org/en/user_manual/python_scripting/introduction.html)
* [Krita Plugin Examples](https://invent.kde.org/graphics/krita-plugins)

---
