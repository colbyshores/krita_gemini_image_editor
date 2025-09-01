Krita Gemini Image Editor
The Krita Gemini Image Editor is a Python plugin that integrates the power of Google's Gemini generative AI directly into your Krita workflow. It allows you to perform complex, prompt-based edits on your images by leveraging the Gemini API to intelligently modify the content of the active layer.

Features
Seamless Integration: Adds a "Gemini Image Editor" action directly into Krita's Tools -> Scripts menu.
Prompt-Based Editing: Use natural language to describe the changes you want to see in your image.
Simple UI: A straightforward dialog lets you input your private Google Gemini API key and your text prompt.
Layer-Specific: The plugin works on the currently active layer, leaving other layers untouched.
Undoable Actions: Each AI-powered edit is registered as a single, atomic action in Krita's history, so you can easily undo it with Ctrl+Z.
How It Works
When you trigger the plugin, it captures the pixel data of the currently selected layer. This image data, along with the text prompt you provide, is sent securely to the Google Gemini API. The API processes the image based on your instructions and returns a new, modified image. The plugin then replaces the original content of your active layer with this new image from Gemini.

Installation
Open Krita.
Navigate to the menu: Settings -> Manage Resources...
In the resource manager window, click the Open Resource Folder button.
This will open your system's file browser to Krita's main resource folder.
From there, navigate into the pykrita subfolder.
Copy the entire gemini_image_editor folder and the gemini_image_editor.desktop file into this pykrita folder.
Restart Krita.
Usage
Open an image and select the layer you wish to edit.
Navigate to the menu: Tools -> Scripts -> Gemini Image Editor...
The Gemini Image Editor dialog will appear.
In the "API Key" field, enter your personal Google Gemini API key.
In the "Prompt" field, enter a description of the changes you want to make (e.g., "make the sky look like a sunset", "add a small boat on the water", "change the art style to watercolor").
Click OK and wait for the operation to complete. The plugin will show status updates in its log window.
Once finished, the active layer will be updated with the new image from Gemini. If you are not happy with the result, you can undo it by pressing Ctrl+Z.
Requirements
Krita (Version 5.0 or newer)
An active internet connection
A valid Google Gemini API key

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
