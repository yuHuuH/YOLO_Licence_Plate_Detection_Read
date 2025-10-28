import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import os
import sys
from collections import Counter
import math
from typing import Tuple, Union
from deskew import determine_skew

def enhance_image(plate_crop):
    enhanced_image = cv2.detailEnhance(plate_crop, sigma_s=10, sigma_r=0.15)
    return enhanced_image

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background) # type: ignore


def get_deskew(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    rotated = rotate(image, angle, (0, 0, 0)) # type: ignore
    # if angle < 10:
    #     return image  
    return rotated

def denoise(img):
    # Apply denoising
    denoised_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Adjust contrast and brightness
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 20    # Brightness control (0-100)
    enhanced_image = cv2.convertScaleAbs(denoised_image, alpha=alpha, beta=beta)
    return enhanced_image

# --- PyInstaller Resource Path Helper ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Model Loading ---
try:
    model = YOLO(resource_path(os.path.join("model", "lp_detect.pt")))
    ocr = YOLO(resource_path(os.path.join("model", "char_detect.pt")))
except Exception as e:
    messagebox.showerror("Model Load Error", f"Could not load YOLO models: {e}")
    sys.exit(1)

# --- getLPText Function ---
def getLPText(plate_crop, plate_ocr):
    """
    Extracts text from a license plate using OCR.
    - If plate is wide (width > height * 2), assume single-row and skip deskew.
    - Otherwise, deskew and apply KMeans-based row separation.
    """
    height, width = plate_crop.shape[:2]
    is_single_row = width > (height * 1.5)
    # plate_crop = enhance_image(plate_crop)
    # if not is_single_row:
    #     plate_crop = get_deskew(plate_crop)

    # Run OCR
    text_plate = plate_ocr(plate_crop, imgsz=640, conf=0.5, verbose=False)
    ocr_pre = text_plate[0]
    id2char = plate_ocr.names
    boxes = ocr_pre.boxes

    if len(boxes) == 0:
        return ""

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    box_coords = boxes.xyxy.cpu().numpy()
    combined = list(zip(box_coords, cls_ids))

    # --- Single-row plate: simple left-to-right sorting ---
    if is_single_row:
        combined.sort(key=lambda x: x[0][0])  # sort by x1
        detected_text = ''.join(id2char[cls_id] for _, cls_id in combined)
        # if len(detected_text) < 8:
        #     plate_crop = get_deskew(plate_crop)
        #     plate_crop = denoise(plate_crop)
        #     text_plate = plate_ocr(plate_crop, imgsz=640, conf=0.5, verbose=False)
        #     ocr_pre = text_plate[0]
        #     boxes = ocr_pre.boxes
        #     cls_ids = boxes.cls.cpu().numpy().astype(int)
        #     box_coords = boxes.xyxy.cpu().numpy()
        #     combined = list(zip(box_coords, cls_ids))
        #     combined.sort(key=lambda x: x[0][0])  # sort by x1
        #     detected_text = ''.join(id2char[cls_id] for _, cls_id in combined)
        #     return detected_text
        return detected_text

    # --- Multi-row plate: use clustering by Y center ---
    # Compute middle line (y-coordinate)
    _, width = plate_crop.shape[:2]
    height = plate_crop.shape[0]
    middle_y = height / 2
    
    # For each character, compute its vertical center
    top_row = []
    bottom_row = []
    for box, cls_id in combined:
        y_center = (box[1] + box[3]) / 2
        x1 = box[0]
        if y_center < middle_y:
            top_row.append((x1, cls_id))
        else:
            bottom_row.append((x1, cls_id))
    
    # Sort each row left-to-right
    top_row.sort(key=lambda item: item[0])
    bottom_row.sort(key=lambda item: item[0])
    
    # Convert class ids to characters
    top_text = ''.join(id2char[cls_id] for _, cls_id in top_row)
    bottom_text = ''.join(id2char[cls_id] for _, cls_id in bottom_row)
    
    # Combine rows as needed (top first, then bottom)
    detected_text = top_text + bottom_text
    return detected_text


# --- Tracking Classes ---
class Track:
    """Represents a single tracked object (license plate)."""
    next_id = 0 # Class-level unique ID counter

    def __init__(self, bbox, lp_text):
        self.track_id = Track.next_id
        Track.next_id += 1
        self.bbox = bbox # [x1, y1, x2, y2]
        self.lp_text_history = [lp_text]
        self.hits = 1 # How many consecutive frames it's been detected
        self.age = 0 # How many frames it has existed
        self.max_history_len = 10 # Number of previous OCR results to consider for smoothing

    def update(self, new_bbox, new_lp_text):
        self.bbox = new_bbox
        self.lp_text_history.append(new_lp_text)
        # Keep history length limited
        if len(self.lp_text_history) > self.max_history_len:
            self.lp_text_history.pop(0)
        self.hits += 1
        self.age = 0 # Reset age since it's been hit

    def predict(self):
        # Use the last known bbox.
        self.age += 1
        return self.bbox

    def get_stabilized_lp_text(self):
        """Returns the most common (stabilized) LP text from history."""
        if not self.lp_text_history:
            return "N/A"
        # Filter out "No text detected" from voting if possible
        clean_history = [text for text in self.lp_text_history if text != ""]
        if not clean_history:
            return "N/A"
        most_common = Counter(clean_history).most_common(1)
        return most_common[0][0] if most_common else "N/A"


class Tracker:
    """Manages multiple tracks."""
    def __init__(self, iou_threshold=0.5, max_age=3, min_hits=3):
        self.tracks = []
        self.iou_threshold = iou_threshold # IoU threshold for associating detections with tracks
        self.max_age = max_age # How many frames a track can go without being detected before deletion
        self.min_hits = min_hits # Minimum hits for a track to be considered 'confirmed'

    def update(self, detections):
        """
        Updates existing tracks with new detections and creates new tracks.
        detections: List of tuples (bbox, lp_text)
        """
        if not self.tracks: # If no existing tracks, create new ones for all detections
            for bbox, lp_text in detections:
                self.tracks.append(Track(bbox, lp_text))
            return self.tracks

        # Predict new positions for existing tracks
        predicted_tracks_bboxes = [track.predict() for track in self.tracks]

        # Initialize lists for matches, unmatched detections, and unmatched tracks
        matches = [] # (track_idx, detection_idx)
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # IoU-based matching
        for d_idx, (d_bbox, _) in enumerate(detections):
            for t_idx, t_bbox in enumerate(predicted_tracks_bboxes):
                if t_idx in unmatched_tracks: # Only consider unmatched tracks
                    iou = self._calculate_iou(d_bbox, t_bbox)
                    if iou > self.iou_threshold:
                        matches.append((t_idx, d_idx))
                        if d_idx in unmatched_detections:
                            unmatched_detections.remove(d_idx)
                        if t_idx in unmatched_tracks:
                            unmatched_tracks.remove(t_idx)
                        break # Move to next detection once a match is found

        # Update matched tracks
        for t_idx, d_idx in matches:
            bbox, lp_text = detections[d_idx]
            self.tracks[t_idx].update(bbox, lp_text)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            bbox, lp_text = detections[d_idx]
            self.tracks.append(Track(bbox, lp_text))

        # Remove old, un-hit tracks
        self.tracks = [track for track in self.tracks if track.age < self.max_age]

        return self.tracks

    def _calculate_iou(self, boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

class LPRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("License Plate Recognition")

        # --- State Variables ---
        self.input_path = None
        self.output_video_path = None
        self.processing_thread = None
        self.stop_processing = False
        self.tracker = Tracker() # Initialize the tracker here

        # --- GUI Elements ---
        self.label_image = tk.Label(master)
        self.label_image.pack(pady=10)

        button_frame = tk.Frame(master)
        button_frame.pack(pady=5)

        self.btn_browse = tk.Button(button_frame, text="Browse File", command=self.browse_file)
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        self.btn_process = tk.Button(button_frame, text="Process File", command=self.start_processing, state=tk.DISABLED)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(button_frame, text="Stop Processing", command=self.stop_current_processing, state=tk.DISABLED, fg="red")
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.result_label = tk.Label(master, text="Detected Text: ")
        self.result_label.pack(pady=5)

        self.status_label = tk.Label(master, text="Status: Ready")
        self.status_label.pack(pady=5)

    # --- File Handling & Display ---
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video File",
            filetypes=[("Media Files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.webm"),
                       ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                       ("Video Files", "*.mp4 *.avi *.mov *.webm"),
                       ("All Files", "*.*")]
        )
        if file_path:
            self.input_path = file_path
            self.result_label.config(text="Detected Text: ")
            self.status_label.config(text=f"Status: Selected {os.path.basename(file_path)}")
            self.btn_process.config(state=tk.NORMAL)
            self.progress_bar['value'] = 0

            if self._is_image_file(file_path):
                self._display_image(file_path)
            else:
                self.label_image.config(image='')
                self.label_image.image = None # type: ignore
                self.label_image.config(text="Video file selected. Click Process to start.")

    def _is_image_file(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

    def _is_video_file(self, path):
        return path.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))

    def _display_image(self, path_or_array):
        if isinstance(path_or_array, str):
            img = cv2.imread(path_or_array)
            if img is None:
                messagebox.showerror("Error", f"Could not read image: {path_or_array}")
                return
        else:
            img = path_or_array

        height, width, _ = img.shape
        max_display_width = 800
        max_display_height = 600

        if width > max_display_width or height > max_display_height:
            scaling_factor = min(max_display_width / width, max_display_height / height)
            img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.label_image.config(image=img_tk, text="")
        self.label_image.image = img_tk # type: ignore

    def _display_image_from_array(self, img_array):
        self._display_image(img_array)

    # --- Processing Control ---
    def start_processing(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        self.stop_processing = False
        self.btn_process.config(state=tk.DISABLED)
        self.btn_browse.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Processing...")
        self.result_label.config(text="Detected Text: ")
        self.progress_bar['value'] = 0
        Track.next_id = 0 # Reset track IDs for a new processing session
        self.tracker = Tracker() # Reset the tracker for a new session

        if self._is_image_file(self.input_path):
            self.processing_thread = threading.Thread(target=self._process_image_file, daemon=True)
        elif self._is_video_file(self.input_path):
            self.output_video_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")],
                title="Save Processed Video As"
            )
            if not self.output_video_path:
                self.reset_gui_state()
                messagebox.showinfo("Processing Cancelled", "Video saving cancelled.")
                return
            self.processing_thread = threading.Thread(target=self._process_video_file, daemon=True)
        else:
            messagebox.showerror("Error", "Unsupported file type.")
            self.reset_gui_state()
            return

        self.processing_thread.start()
        self.master.after(100, self._check_processing_thread)

    def _check_processing_thread(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.master.after(100, self._check_processing_thread)
        else:
            self.reset_gui_state()
            if not self.stop_processing:
                self.status_label.config(text="Status: Done!")
            else:
                self.status_label.config(text="Status: Processing Stopped.")

    def stop_current_processing(self):
        self.stop_processing = True
        self.status_label.config(text="Status: Stopping...")
        self.btn_stop.config(state=tk.DISABLED)

    def reset_gui_state(self):
        self.btn_process.config(state=tk.NORMAL if self.input_path else tk.DISABLED)
        self.btn_browse.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0

    # --- Image Processing Logic ---
    def _process_image_file(self):
        try:
            img = cv2.imread(self.input_path) # type: ignore
            if img is None:
                self.master.after(0, lambda: messagebox.showerror("Error", f"Could not read image: {self.input_path}"))
                return

            results = model(img, imgsz=640, verbose=False, conf=0.5)

            img_with_boxes_and_text = img.copy()
            detected_lps = []

            for box in results[0].boxes:
                if self.stop_processing: break
                class_id = int(box.cls[0])
                if class_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
                    plate_crop = img[y1:y2, x1:x2]

                    if plate_crop.shape[0] == 0 or plate_crop.shape[1] == 0:
                        continue

                    final_lp = getLPText(plate_crop, ocr)
                    detected_lps.append(final_lp)

                    cv2.rectangle(img_with_boxes_and_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 3
                    text_color = (0, 255, 0)

                    (text_width, text_height), baseline = cv2.getTextSize(final_lp, font, font_scale, font_thickness)
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 5

                    cv2.putText(img_with_boxes_and_text, final_lp, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            self.master.after(0, lambda: self._display_image_from_array(img_with_boxes_and_text))
            self.master.after(0, lambda: self.result_label.config(text="Detected Text: " + ", ".join(detected_lps)))
            self.master.after(0, lambda: self.progress_bar.config(value=100))

        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred during image processing: {e}"))
        finally:
            self.master.after(100, self._check_processing_thread)

    # --- Video Processing Logic (with tracking) ---
    def _process_video_file(self):
        cap = cv2.VideoCapture(self.input_path) # type: ignore
        if not cap.isOpened():
            self.master.after(0, lambda: messagebox.showerror("Error", f"Could not open video file: {self.input_path}"))
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = 0
        if self.output_video_path.lower().endswith(".mp4"): # type: ignore
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        elif self.output_video_path.lower().endswith(".avi"): # type: ignore
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
        else:
            self.master.after(0, lambda: messagebox.showerror("Error", "Unsupported output video format. Please save as .mp4 or .avi."))
            cap.release()
            return

        out = None
        if self.output_video_path:
            try:
                out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
            except Exception as e:
                self.master.after(0, lambda: messagebox.showerror("Video Writer Error", f"Could not create video writer: {e}\nTry changing output format or codec."))
                cap.release()
                return

        frame_count = 0
        all_unique_lps = [] # To store all unique and confirmed LPs detected across the video

        try:
            while cap.isOpened() and not self.stop_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Run LP Detection
                results = model(frame, imgsz=640, verbose=False, conf=0.5)
                current_detections = [] # Format: [(bbox, lp_text), ...]

                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    if class_id == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                        plate_crop = frame[y1:y2, x1:x2]

                        if plate_crop.shape[0] == 0 or plate_crop.shape[1] == 0:
                            continue

                        lp_text = getLPText(plate_crop, ocr)
                        current_detections.append(([x1, y1, x2, y2], lp_text))

                # 2. Update Tracker with current detections
                self.tracker.update(current_detections)

                # 3. Prepare frame for display/saving with tracked results
                img_with_boxes_and_text = frame.copy()
                current_frame_displayed_lps = [] # LPs that are confirmed and displayed on current frame

                for track in self.tracker.tracks:
                    # Only draw/use confirmed tracks (those hit enough times)
                    if track.hits >= self.tracker.min_hits:
                        x1, y1, x2, y2 = map(int, track.bbox)
                        stabilized_lp_text = track.get_stabilized_lp_text()

                        # Only accept if the number of characters on the license plate is 8 or 9
                        if stabilized_lp_text != "N/A" and (len(stabilized_lp_text) == 8 or len(stabilized_lp_text) == 9):
                            # Add to overall unique list if new and confirmed
                            if stabilized_lp_text not in all_unique_lps:
                                all_unique_lps.append(stabilized_lp_text)

                            current_frame_displayed_lps.append(stabilized_lp_text)

                            cv2.rectangle(img_with_boxes_and_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            font_thickness = 3
                            text_color = (0, 255, 0)
                            (text_width, text_height), baseline = cv2.getTextSize(stabilized_lp_text, font, font_scale, font_thickness)
                            text_x = x1
                            text_y = y1 - 5 if y1 - 5 > text_height else y1 + text_height + 5
                            cv2.putText(img_with_boxes_and_text, stabilized_lp_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                if out:
                    out.write(img_with_boxes_and_text)

                frame_count += 1
                progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0

                self.master.after(0, lambda f=img_with_boxes_and_text, p=progress, lps=current_frame_displayed_lps:
                                  self._update_video_gui_elements(f, p, lps))

            if self.stop_processing:
                self.master.after(0, lambda: self.status_label.config(text="Status: Processing Stopped."))
            else:
                self.master.after(0, lambda: self.status_label.config(text="Status: Video Processing Complete."))
                if all_unique_lps:
                    self.master.after(0, lambda: self._prompt_save_detected_lps(all_unique_lps))

        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred during video processing: {e}"))
        finally:
            cap.release()
            if out:
                out.release()

    def _update_video_gui_elements(self, frame, progress, current_frame_lps):
        self._display_image_from_array(frame)
        self.progress_bar['value'] = progress
        self.result_label.config(text="Last Frame Confirmed LPs: " + (", ".join(current_frame_lps) if current_frame_lps else "N/A"))
        self.master.update_idletasks()

    def _prompt_save_detected_lps(self, lps_list):
        if messagebox.askyesno("Save Detected LPs", "Do you want to save all unique detected license plates to a text file?"):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save All Unique Detected License Plates As"
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        for lp in lps_list:
                            f.write(lp + '\n')
                    messagebox.showinfo("Save Complete", f"All unique detected license plates saved to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Could not save file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LPRecognitionApp(root)
    root.mainloop()
