# Krita Gemini Image Editor

The **Krita Gemini Image Editor** is a Python plugin that integrates the power of Google's Gemini generative AI directly into your Krita workflow. It allows you to perform complex, prompt-based edits on your images and even generate short animations, all from a convenient tabbed interface within Krita.

-----

## ✨ Features

  * **Tabbed Interface:** A clean UI separates functionality into two main tabs: "Stills (Preview)" and "Animation".
  * **Still Image Generation:** Use natural language to describe changes to your image on the active layer.
  * **Preview History:** The "Stills" tab maintains a visual history of your generated images. Simply click a thumbnail to revert the active layer to a previous version.
  * **Multi-Frame Animation:** Generate a sequence of frames based on a starting image and a prompt. The plugin creates new layers for each frame and attempts to set up an animation timeline.
  * **Motion Guides:** Use optional motion guides to give the AI hints about the desired direction of movement in an animation.
  * **Background Processing:** Animation generation runs in a background thread, keeping the Krita UI responsive and showing a progress bar for long operations.
  * **Seamless Integration:** Adds a "Gemini Image Editor" action directly into Krita's `Tools` -> `Scripts` menu.

-----

## ⚙️ How It Works

When you trigger the plugin, it captures the pixel data of the currently selected layer.

*   **In the "Stills" tab:** This image data, along with your text prompt, is sent to the Google Gemini API. The API returns a new, modified image. The plugin replaces the content of your active layer with this new image and adds a thumbnail to the preview queue.
*   **In the "Animation" tab:** The plugin uses the initial image and prompt to generate the first new frame. It then repeatedly sends the *last generated frame* back to the API to create the next one in sequence. Once all frames are generated, it creates a new layer group in your document, adds each frame as a separate layer, and attempts to trigger Krita's "Convert Layers to Frames" action to place them on the animation timeline.

-----

## ⬇️ Installation

1.  Open Krita.
2.  Navigate to the menu: `Settings` -> `Manage Resources...`
3.  In the resource manager window, click the `Open Resource Folder` button.
4.  This will open your system's file browser to Krita's main resource folder.
5.  From there, navigate into the `pykrita` subfolder.
6.  Copy the entire `gemini_image_editor` folder and the `gemini_image_editor.desktop` file into this `pykrita` folder.
7.  Restart Krita.

-----

## 🚀 Usage

1.  Open an image and select the layer you wish to edit.
2.  Navigate to the menu: `Tools` -> `Scripts` -> `Gemini Image Editor...`
3.  The Gemini Image Editor dialog will appear.
4.  In the **API Key** field, enter your personal Google Gemini API key. This key is saved for future sessions.

### Using the "Stills (Preview)" Tab

This tab is for generating and comparing single image variations.

1.  Enter a description of the changes you want to make in the **Prompt** field (e.g., "make the sky look like a sunset", "add a small boat on the water").
2.  Click **Generate Still**.
3.  The active layer will be updated with the new image from Gemini. A thumbnail of the result will be added to the **Preview Queue** on the right.
4.  You can generate more variations with different prompts. To switch back to a previous version, simply click its thumbnail in the queue.

### Using the "Animation" Tab

This tab is for generating a sequence of frames for an animation.

1.  Enter a prompt that describes the animation (e.g., "a candle flame flickering gently", "a ball bouncing across the screen").
2.  Set the number of **Frames** you want to generate and the desired **FPS** (Frames Per Second) for the final animation.
3.  **Start animation from current image:** Keep this checked if you want the animation to evolve from your currently selected layer. The total animation will be `Frames` long. If you uncheck it, the plugin will generate all frames from scratch based on the prompt.
4.  **(Optional) Use motion guides:** Check this to provide a simple, predefined motion path to the AI. This is useful for guiding objects across the frame.
5.  Click **Generate Animation**. A progress dialog will appear.
6.  Once finished, the plugin will create a new layer group containing all the generated frames and attempt to convert them into a playable animation on Krita's timeline. If you are not happy with the result, you can undo the entire operation by pressing `Ctrl+Z`.

-----

## ✅ Requirements

  * Krita (Version 5.0 or newer)
  * An active internet connection
  * A valid Google Gemini API key

-----

## 📦 Krita Plugin Packaging Guide

This section provides instructions on how to properly package and distribute a Krita Python plugin. It has been tested using **Krita 5.0.2 on Pop!_OS 22.04** (an Ubuntu-based Linux distribution).

-----

### 1\. Directory Structure

Your plugin should be a zip file and have the following layout:

```
./
├── gemini_image_editor.desktop
├── gemini_image_editor/
│   ├── __init__.py
│   └── gemini_image_editor.py
└── README.md
└── LICENSE.md
```

**Key files:**

  * `gemini_image_editor.desktop` → The metadata file used by Krita to display the plugin.
  * `__init__.py` → Marks the `gemini_image_editor` folder as a Python module.
  * `gemini_image_editor.py` → Contains the main plugin logic.

-----

### 2. Installing the Plugin

On **Pop\!\_OS 22.04 (or Ubuntu-based systems)**, Krita plugins are installed under:

```
~/.local/share/krita/pykrita/
```

To install:

```bash
mkdir -p ~/.local/share/krita/pykrita
cp -r gemini_image_editor ~/.local/share/krita/pykrita/
```

Restart Krita, then:

1.  Go to `Settings` -\> `Configure Krita` -\> `Python Plugins`.
2.  Enable your plugin. 
3.  It will now appear under `Tools` -> `Scripts` -> `Gemini Image Editor`.

-----

### 3. Distributing the Plugin

For distribution:

1.  Zip your plugin directory:

    ```bash
    zip -r gemini_image_editor.zip my_plugin/
    ```

2.  Share `gemini_image_editor.zip`. Users can unzip and place it inside their Krita `pykrita` folder.

-----

## 🧪 Tested Environment

  * **Krita:** 5.0.2
  * **OS:** Pop!_OS 22.04 (Ubuntu-based Linux)

Functionality and packaging have been confirmed to be working in this setup. Behavior on other systems or newer versions of Krita may vary.

-----

## 📚 References

  * [Krita Python Scripting Documentation](https://docs.krita.org/en/user_manual/python_scripting/introduction.html)
  * [Krita Plugin Examples](https://invent.kde.org/graphics/krita-plugins)
