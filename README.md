# License Plate Detection & Read

## Introduction

This project uses **YOLOv12** to detect and read Vietnamese license plates.  
It supports both images and videos, with optional saving for processed videos.

## Models

This project contains 2 trained Yolo12 models:

```file
project/
└── final/
    └── model/
        ├── lp_detect.pt
        └── char_detect.pt
```

- **lp_detect.pt** → Detects license plates in images or frames  
- **char_detect.pt** → Reads characters from detected plates  

If you want to use another trained OCR model, such as **Automated License Plate Recognition (ALPR)**, you can use the [`fast_alpr`](https://pypi.org/project/fast-alpr/) library:

```python
from fast_alpr import ALPR
import os

alpr = ALPR(ocr_model="global-plates-mobile-vit-v2-model")
```

## Workflow

Detect -> Crop & Read -> Return result
**Input**: The input and be both image or video:\

- Images and videos will be processed directly, with videos have option to save as .mp4 for review later

## Usage

Fist you need to install required python package with requirement.txt

```python
pip install -r requirements.txt
```

After you have installed the required package open the file that contain `model` folder 

```terminal
cd your_path\YOLO Licence Plate Detection & Read\final
```

Then run the app.py 

```terminal
python app.py
```

or

```terminal
python <your_path>\YOLO Licence Plate Detection & Read\final\app.py
```
