# License Plate Detection & Recognition

This repository provides a desktop GUI application (Tkinter) that detects and recognizes Vietnamese license plates using YOLO object detection and a YOLO-based character recognizer.

Key features
- Graphical application (Tkinter) with image preview and progress UI
- Detects license plates in images and video frames using an ultralytics YOLO detector
- Per-plate OCR using a second YOLO model (characters as classes)
- Simple tracking and result stabilization across frames
- Optionally save annotated video and export unique detected license plates to a text file

Files of interest
- [final/app.py](<D:/FPT/Git test/YOLO_Licence_Plate_Detection_Read.worktrees/update-md-professionalism/final/app.py>) — main GUI application and processing logic
- [final/Model/](<D:/FPT/Git test/YOLO_Licence_Plate_Detection_Read.worktrees/update-md-professionalism/final/Model/>) — expected model files (see below)
- [requirements.txt](<D:/FPT/Git test/YOLO_Licence_Plate_Detection_Read.worktrees/update-md-professionalism/requirements.txt>) — Python dependencies
- [example/input/](<D:/FPT/Git test/YOLO_Licence_Plate_Detection_Read.worktrees/update-md-professionalism/example/input/>) — example image and video for quick testing
- [example/output/](<D:/FPT/Git test/YOLO_Licence_Plate_Detection_Read.worktrees/update-md-professionalism/example/output/>) — example outputs

Models and where they are loaded
- The application loads YOLO models using the ultralytics package from the application resource path. The code attempts to load:
  - model/op_lp_detect.pt (detector)
  - model/char_detect.pt (OCR/character detector)

On Windows the repository includes models under final\Model (capital M). The app uses a resource helper to support both running from source and from a PyInstaller bundle.

Included model files (in this repo)
- final/Model/op_lp_detect.pt
- final/Model/lp_detect.pt
- final/Model/char_detect.pt

Requirements
- Python 3.8+ recommended
- Install dependencies:

```
pip install -r requirements.txt
```

requirements.txt includes (representative):
- opencv-python
- numpy
- ultralytics
- scikit-learn
- Pillow
- deskew

Notes: If you plan to use GPU acceleration, install the appropriate torch + CUDA build compatible with your hardware and the ultralytics / YOLO requirements.

How the application works (behavior matched to code)
- Launch: run the GUI with:

```python
final\app.py
```

- Browse File: click "Browse File" to select an image or a video (supported extensions: jpg, jpeg, png, bmp, mp4, avi, mov, webm).
- Process File: for images the app runs detection and OCR and immediately shows annotated image and detected plate strings in the GUI.
- For video files the app prompts for an output path (asks where to save the processed video). The app then processes frames, runs the detector and OCR, performs simple IoU-based tracking, and stabilizes OCR results per-track.
- When video processing finishes the app offers to save all unique detected license plates to a text file.

Implementation details pulled from the code
- Model inference parameters: imgsz=640, conf=0.5 (used for both detector and OCR calls)
- OCR: the char-detection model returns per-character class ids; the app maps class ids to characters and sorts characters left-to-right (or by row for multi-row plates)
- Deskew helper: deskew.determine_skew is imported and used in rotate/deskew helper functions (deskew is required)
- Tracking & stabilization:
  - Tracker class uses IoU matching and Track objects to track plates across frames
  - Default tracker thresholds in code: iou_threshold=0.5, max_age=3, min_hits=3
  - A track is considered "confirmed" when hits >= min_hits (default 3)
  - Stabilized plate text is the most-common string from recent OCR results for that track
  - The app only treats stabilized strings of length 8 or 9 characters as valid when adding to the unique list (this is implemented in the video loop)

Input and output expectations
- Input: image files (.jpg/.png/.bmp) or video files (.mp4/.avi/.mov/.webm)
- Interactive output: annotated image preview for single images; live annotated frames and progress bar for video
- Saved output (video): user-specified MP4/AVI (the writer chooses codec based on extension)
- Saved LP list: user can save a .txt file listing all unique, confirmed license plates detected during a video run

Examples (quick test)
- Process the repository example image with the GUI: run the app and choose
  - [example/input/image.jpg](<D:/FPT/Git test/YOLO_Licence_Plate_Detection_Read.worktrees/update-md-professionalism/example/input/image.jpg>)
- Process the repository example video with the GUI and save annotated output to [example/output/video.mp4](<D:/FPT/Git test/YOLO_Licence_Plate_Detection_Read.worktrees/update-md-professionalism/example/output/video.mp4>)
