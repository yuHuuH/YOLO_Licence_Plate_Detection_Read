# License Plate Detection & Recognition

A desktop application for **Vietnamese license plate detection and recognition** using YOLO-based object detection and character recognition. The application provides a Tkinter graphical interface for processing images and videos, with support for license plate tracking, OCR stabilization, annotated output, and license plate export.

## Features

* **License Plate Detection**

  * Detects Vietnamese license plates in images and video frames using an Ultralytics YOLO model.
* **Character Recognition**

  * Performs per-plate OCR using a second YOLO model trained to detect individual characters.
  * Reconstructs license plate text by ordering detected characters spatially.
  * Supports both single-row and multi-row license plates.
* **Video Tracking**

  * Tracks detected license plates across consecutive frames using IoU-based matching.
  * Stabilizes OCR results across frames to reduce recognition noise.
* **Graphical User Interface**

  * Built with Tkinter.
  * Provides file selection, image preview, video-processing progress, and result display.
* **Annotated Output**

  * Displays detected license plates and recognized text on processed images and video frames.
  * Supports saving processed videos to a user-selected output path.
* **License Plate Export**

  * Exports unique, confirmed license plate numbers detected during video processing to a `.txt` file.

---

## Project Structure

```text
.
├── final/
│   ├── app.py
│   └── Model/
│       ├── op_lp_detect.pt
│       ├── lp_detect.pt
│       └── char_detect.pt
│
├── example/
│   ├── input/
│   │   ├── image.jpg
│   │   └── video.mp4
│   │
│   └── output/
│       └── video.mp4
│
└── requirements.txt
```

### Important Files

| File / Directory   | Description                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| `final/app.py`     | Main Tkinter application, detection, OCR, tracking, and video-processing logic |
| `final/Model/`     | Directory containing the YOLO model weights                                    |
| `op_lp_detect.pt`  | License plate detection model used by the application                          |
| `char_detect.pt`   | Character detection model used for OCR                                         |
| `lp_detect.pt`     | Additional license plate model included in the repository                      |
| `requirements.txt` | Python dependencies                                                            |
| `example/input/`   | Sample image and video for testing                                             |
| `example/output/`  | Example processed output                                                       |

> **Note:** The default application pipeline loads `op_lp_detect.pt` and `char_detect.pt`. The included `lp_detect.pt` model is not loaded by the default inference workflow.

---

## Requirements

### Software

* Python **3.8+**
* Tkinter
* OpenCV
* Ultralytics
* PyTorch

### Python Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

The project includes dependencies such as:

* `opencv-python`
* `numpy`
* `ultralytics`
* `scikit-learn`
* `Pillow`

For GPU acceleration, install a compatible **PyTorch + CUDA** build for your hardware and ensure it is compatible with the installed Ultralytics environment.

---

## Model Files

The application expects the following model files:

```text
final/
└── Model/
    ├── op_lp_detect.pt
    └── char_detect.pt
```

The application includes resource-path handling to support locating model files when running:

* Directly from the source repository
* From a PyInstaller-generated application bundle

### Model Roles

| Model             | Purpose                                                   |
| ----------------- | --------------------------------------------------------- |
| `op_lp_detect.pt` | Detects license plate regions                             |
| `char_detect.pt`  | Detects individual characters inside license plates       |
| `lp_detect.pt`    | Additional license plate model included in the repository |

---

## How It Works

The application uses a two-stage YOLO pipeline:

```text
Input Image / Video
        │
        ▼
License Plate Detection
        │
        ▼
Plate Region Cropping
        │
        ▼
Character Detection
        │
        ▼
Character Class Mapping
        │
        ▼
Character Ordering
        │
        ▼
License Plate Text
```

For video input, an additional tracking and stabilization stage is applied:

```text
Video Frames
     │
     ▼
License Plate Detection
     │
     ▼
IoU-Based Tracking
     │
     ▼
Per-Plate Character Recognition
     │
     ▼
OCR Result Stabilization
     │
     ▼
Unique License Plate Collection
```

---

## Detection

The application uses `op_lp_detect.pt` to detect license plate regions.

Default inference parameters:

```text
Image size: 640
Confidence threshold: 0.5
```

The same inference configuration is used for the character detection model.

---

## Character Recognition

Each detected license plate is cropped and passed to the `char_detect.pt` model.

The character detector returns individual character detections containing:

* Character class ID
* Bounding box
* Confidence score

The application maps each class ID to its corresponding character and sorts the detected characters spatially to reconstruct the license plate string.

For multi-row license plates, character positions are analyzed by row before determining the final reading order.

---

## Video Tracking

Video processing uses a lightweight IoU-based tracker to associate license plate detections across consecutive frames.

Default tracker parameters:

| Parameter         |    Default |
| ----------------- | ---------: |
| IoU threshold     |      `0.5` |
| Maximum track age | `3` frames |
| Minimum hits      |        `3` |

A track becomes **confirmed** after receiving at least three successful detections.

This allows the application to maintain a consistent identity for the same license plate across multiple frames.

---

## OCR Stabilization

OCR predictions may vary between frames because of:

* Motion blur
* Lighting changes
* Occlusion
* Camera movement
* Small or low-resolution plates
* Temporary detection errors

To reduce these variations, the application stores recent OCR results for each tracked plate and selects the **most frequently occurring result** as the stabilized license plate number.

For example:

```text
Frame 1 → 51A12345
Frame 2 → 51A12345
Frame 3 → 51A12345
Frame 4 → 51A1234S
Frame 5 → 51A12345
```

The stabilized result becomes:

```text
51A12345
```

During video processing, only stabilized strings with **8 or 9 characters** are considered valid for the unique license plate collection.

## Usage

### 1. Install Dependencies

Clone or download the repository and install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Make sure the required model files are located at:

```text
final/Model/op_lp_detect.pt
final/Model/char_detect.pt
```

### 3. Launch the Application

From the project root:

```bash
python final/app.py
```

### 4. Select an Input File

Click **Browse File** and select an image or video.

### Supported Image Formats

```text
.jpg
.jpeg
.png
.bmp
```

### Supported Video Formats

```text
.mp4
.avi
.mov
.webm
```

---

## Image Processing

For image input:

1. Click **Browse File**.
2. Select an image.
3. Start processing.
4. The application detects license plates.
5. Each detected plate is passed to the character recognition model.
6. The annotated image and recognized plate numbers are displayed in the GUI.

---

## Video Processing

For video input:

1. Click **Browse File**.
2. Select a supported video.
3. Choose an output location when prompted.
4. The application processes the video frame by frame.
5. License plates are detected and tracked.
6. Character recognition is performed for each detected plate.
7. OCR results are stabilized across frames.
8. The annotated video is saved to the selected output path.
9. After processing, the application can export unique detected license plates to a `.txt` file.

---

## Example

The repository contains sample input files for quick testing.

### Example Image

```text
example/input/image.jpg
```

Run the application and select the image through **Browse File**.

### Example Video

```text
example/input/video.mp4
```

The processed video can be saved to:

```text
example/output/video.mp4
```

The application allows the output location to be selected through the save dialog.

---

## Output

### Image Output

For image processing, the application provides:

* Annotated image preview
* License plate bounding boxes
* Recognized license plate text

### Video Output

For video processing, the application provides:

* Annotated video frames
* License plate tracking
* OCR results
* Processing progress
* Saved output video
* Optional unique license plate list

Example license plate export:

```text
51A12345
59B67890
62C123456
```

The actual results depend on the input media and model performance.

---

## Performance Considerations

Processing performance depends on:

* CPU/GPU hardware
* PyTorch configuration
* CUDA availability
* Input resolution
* Video frame rate
* Number of license plates per frame
* YOLO model architecture
* Image quality

Recognition accuracy can also be affected by:

* Motion blur
* Poor lighting
* Extreme viewing angles
* Occlusion
* Low-resolution license plates
* Reflections
* Dirty or damaged plates

For faster inference, a compatible NVIDIA GPU with CUDA-enabled PyTorch can be used.

---

## Troubleshooting

### Model Not Found

Verify that the required files exist:

```text
final/Model/op_lp_detect.pt
final/Model/char_detect.pt
```

If running a packaged application, verify that the model files were included in the PyInstaller build and that the resource-path logic points to the correct location.

### Dependency Errors

Reinstall the project dependencies:

```bash
pip install -r requirements.txt
```

If you encounter PyTorch or CUDA errors, verify that your PyTorch installation is compatible with your GPU, CUDA environment, Python version, and Ultralytics version.

### Poor Detection or OCR Results

Recognition performance may decrease when license plates are:

* Too small
* Blurred
* Severely rotated
* Partially occluded
* Poorly illuminated
* Overexposed or underexposed

Improving input quality or retraining/fine-tuning the YOLO models with a more representative dataset may improve results.

---

## Technology Stack

| Technology           | Purpose                                   |
| -------------------- | ----------------------------------------- |
| **Python**           | Application development                   |
| **Ultralytics YOLO** | License plate and character detection     |
| **PyTorch**          | Deep learning inference                   |
| **OpenCV**           | Image and video processing                |
| **Tkinter**          | Desktop graphical interface               |
| **NumPy**            | Numerical processing                      |
| **Pillow**           | Image handling and GUI integration        |
| **scikit-learn**     | Supporting machine-learning functionality |

---

## Tags & Keywords

### Core Technologies

`Python` · `YOLO` · `Ultralytics` · `PyTorch` · `OpenCV` · `Tkinter`

### Computer Vision

`Computer Vision` · `Object Detection` · `OCR` · `Character Recognition` · `Image Processing` · `Object Tracking` · `Video Processing`

### Application Domain

`License Plate Detection` · `License Plate Recognition` · `Automatic License Plate Recognition` · `ALPR` · `ANPR` · `Vietnamese License Plates`

### Machine Learning

`Deep Learning` · `Machine Learning` · `YOLO Object Detection` · `License Plate OCR` · `Real-Time Detection` · `Video Analytics`

## License

This project is licensed under the MIT License.

## Disclaimer

This project is intended for **research, educational, and demonstration purposes**. License plate recognition performance depends on the quality of the input data and the models used. Ensure that any deployment complies with applicable privacy, data protection, and local regulations concerning the collection and processing of vehicle identification information.
