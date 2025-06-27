import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog, messagebox, ttk # ttk for themed widgets like Progressbar
from PIL import Image, ImageTk
import threading # To run heavy processing in a separate thread
import os # For path manipulation and checking file extensions
import sys # For PyInstaller path handling

# --- PyInstaller Resource Path Helper ---
# This function helps locate bundled files (like your models) when running as an .exe
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Model Loading ---
# Load models globally using the resource_path helper
try:
    # Assuming 'model' folder is at the same level as your script
    model = YOLO(resource_path(os.path.join("model", "lp_detect.pt")))
    ocr = YOLO(resource_path(os.path.join("model", "char_detect.pt")))
except Exception as e:
    messagebox.showerror("Model Load Error", f"Could not load YOLO models: {e}")
    sys.exit(1) # Exit the application if models cannot be loaded

# --- getLPText Function (remains the same) ---
def getLPText(plate_crop, plate_ocr):
    """
    Extracts text from a license plate using OCR after correcting skew.
    """
    # Run OCR
    text_plate = plate_ocr(plate_crop, imgsz=640, conf=0.5, verbose=False)
    ocr_pre = text_plate[0]
    id2char = plate_ocr.names
    boxes = ocr_pre.boxes

    if len(boxes) == 0:
        return "No text detected"

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    box_coords = boxes.xyxy.cpu().numpy()
    combined = list(zip(box_coords, cls_ids))

    # Cluster based on vertical (Y) position
    y_centers = np.array([[((box[1] + box[3]) / 2)] for box, _ in combined])
    # Ensure n_clusters is not greater than the number of samples
    n_clusters = min(2, len(combined))

    if n_clusters == 0:
        return "No text detected for clustering"

    # Handle cases where KMeans might fail with too few samples or specific configurations
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(y_centers)
    except Exception as e:
        # Fallback if KMeans fails, e.g., if n_clusters=2 but only 1 sample
        # Or if the number of features (1 for y_centers) is too small for default KMeans init
        print(f"KMeans clustering failed: {e}. Attempting single row processing.")
        labels = np.zeros(len(combined), dtype=int) # Treat all as one cluster

    # Group by row clusters
    rows = [[] for _ in range(n_clusters)]
    for (box, cls_id), label in zip(combined, labels):
        x1 = box[0]
        rows[label].append((x1, cls_id))

    # Sort rows top-to-bottom
    # Only sort if there's more than one cluster
    if n_clusters > 1:
        row_avg_y = [np.mean([y_centers[i][0] for i in range(len(labels)) if labels[i] == r]) for r in range(n_clusters)]
        sorted_rows = [row for _, row in sorted(zip(row_avg_y, rows), key=lambda x: x[0])]
    else:
        sorted_rows = rows # If only one cluster, no need to sort rows by Y

    # Sort each row left-to-right
    detected_text = []
    for row in sorted_rows:
        row.sort(key=lambda item: item[0])  # sort by x
        row_text = ''.join(id2char[cls_id] for _, cls_id in row)
        detected_text.append(row_text)

    final_text = ''.join(detected_text)
    return final_text


class LPRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("License Plate Recognition")

        # --- State Variables ---
        self.input_path = None # Stores the path to the selected image or video file
        self.output_video_path = None # Stores the path where processed video will be saved
        self.processing_thread = None # Holds the thread for background processing
        self.stop_processing = False # Flag to signal the processing thread to stop

        # --- GUI Elements ---
        # Label to display the image/video frame
        self.label_image = tk.Label(master)
        self.label_image.pack(pady=10)

        # Frame for buttons
        button_frame = tk.Frame(master)
        button_frame.pack(pady=5)

        self.btn_browse = tk.Button(button_frame, text="Browse File", command=self.browse_file)
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        self.btn_process = tk.Button(button_frame, text="Process File", command=self.start_processing, state=tk.DISABLED)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(button_frame, text="Stop Processing", command=self.stop_current_processing, state=tk.DISABLED, fg="red")
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Progress bar for video processing
        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Label to display detected text
        self.result_label = tk.Label(master, text="Detected Text: ")
        self.result_label.pack(pady=5)

        # Label to display current status
        self.status_label = tk.Label(master, text="Status: Ready")
        self.status_label.pack(pady=5)

    # --- File Handling & Display ---
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video File",
            filetypes=[("Media Files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov"),
                       ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                       ("Video Files", "*.mp4 *.avi *.mov"),
                       ("All Files", "*.*")]
        )
        if file_path:
            self.input_path = file_path
            self.result_label.config(text="Detected Text: ")
            self.status_label.config(text=f"Status: Selected {os.path.basename(file_path)}")
            self.btn_process.config(state=tk.NORMAL)
            self.progress_bar['value'] = 0

            # If it's an image, display it immediately
            if self._is_image_file(file_path):
                self._display_image(file_path)
            else:
                # Clear image display if it was previously an image, or for video
                self.label_image.config(image='')
                self.label_image.image = None
                self.label_image.config(text="Video file selected. Click Process to start.")


    def _is_image_file(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))

    def _is_video_file(self, path):
        return path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')) # Added .webm for broader support

    def _display_image(self, path_or_array):
        """Displays an image (from path or numpy array) in the GUI."""
        if isinstance(path_or_array, str): # It's a file path
            img = cv2.imread(path_or_array)
            if img is None:
                messagebox.showerror("Error", f"Could not read image: {path_or_array}")
                return
        else: # It's a numpy array (image frame)
            img = path_or_array

        # Resize image to fit in the GUI display area
        height, width, _ = img.shape
        max_display_width = 800
        max_display_height = 600

        if width > max_display_width or height > max_display_height:
            scaling_factor = min(max_display_width / width, max_display_height / height)
            img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.label_image.config(image=img_tk, text="") # Clear text if image is shown
        self.label_image.image = img_tk # Keep a reference to prevent garbage collection!

    def _display_image_from_array(self, img_array):
        """Wrapper to display an image numpy array."""
        self._display_image(img_array)

    # --- Processing Control ---
    def start_processing(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        self.stop_processing = False # Reset stop flag
        self.btn_process.config(state=tk.DISABLED)
        self.btn_browse.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Processing...")
        self.result_label.config(text="Detected Text: ")
        self.progress_bar['value'] = 0

        if self._is_image_file(self.input_path):
            self.processing_thread = threading.Thread(target=self._process_image_file, daemon=True) # daemon=True allows thread to exit with main app
        elif self._is_video_file(self.input_path):
            self.output_video_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")],
                title="Save Processed Video As"
            )
            if not self.output_video_path: # User cancelled save dialog
                self.reset_gui_state()
                messagebox.showinfo("Processing Cancelled", "Video saving cancelled.")
                return
            self.processing_thread = threading.Thread(target=self._process_video_file, daemon=True)
        else:
            messagebox.showerror("Error", "Unsupported file type.")
            self.reset_gui_state()
            return

        self.processing_thread.start()
        # Periodically check if the thread is alive to update GUI when done
        self.master.after(100, self._check_processing_thread)

    def _check_processing_thread(self):
        """Checks the processing thread's status and updates GUI accordingly."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.master.after(100, self._check_processing_thread) # Check again after 100ms
        else:
            # Processing is done or stopped
            self.reset_gui_state()
            if not self.stop_processing: # Only show 'Done' if not stopped by user
                self.status_label.config(text="Status: Done!")
            else:
                self.status_label.config(text="Status: Processing Stopped.")


    def stop_current_processing(self):
        """Sets the flag to stop ongoing processing."""
        self.stop_processing = True
        self.status_label.config(text="Status: Stopping...")
        self.btn_stop.config(state=tk.DISABLED) # Disable stop button while it's stopping

    def reset_gui_state(self):
        """Resets GUI elements to their default state after processing."""
        self.btn_process.config(state=tk.NORMAL if self.input_path else tk.DISABLED)
        self.btn_browse.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        # self.status_label.config(text="Status: Ready") # Let _check_processing_thread set final status


    # --- Image Processing Logic ---
    def _process_image_file(self):
        """Handles processing for an image file."""
        try:
            img = cv2.imread(self.input_path)
            if img is None:
                self.master.after(0, lambda: messagebox.showerror("Error", f"Could not read image: {self.input_path}"))
                return

            results = model(img, imgsz=640, verbose=False)

            img_with_boxes_and_text = img.copy()
            detected_lps = []

            for box in results[0].boxes:
                if self.stop_processing: break # Allow stopping mid-image (though less critical for images)
                class_id = int(box.cls[0])
                if class_id == 0: # Assuming class_id 0 is for license plates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Ensure coordinates are within image bounds
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)

                    plate_crop = img[y1:y2, x1:x2]

                    if plate_crop.shape[0] == 0 or plate_crop.shape[1] == 0:
                        continue # Skip empty or invalid crops

                    final_lp = getLPText(plate_crop, ocr)
                    detected_lps.append(final_lp)

                    # Draw the bounding box
                    cv2.rectangle(img_with_boxes_and_text, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put the detected text on the image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 3
                    text_color = (0, 255, 0)

                    (text_width, text_height), baseline = cv2.getTextSize(final_lp, font, font_scale, font_thickness)
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 5

                    cv2.putText(img_with_boxes_and_text, final_lp, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Update GUI from the main thread
            self.master.after(0, lambda: self._display_image_from_array(img_with_boxes_and_text))
            self.master.after(0, lambda: self.result_label.config(text="Detected Text: " + ", ".join(detected_lps)))
            self.master.after(0, lambda: self.progress_bar.config(value=100))

        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred during image processing: {e}"))
        finally:
            # Reset GUI state after a short delay to ensure updates propagate
            self.master.after(100, self._check_processing_thread)


    # --- Video Processing Logic ---
    def _process_video_file(self):
        """Handles processing for a video file, including saving output."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.master.after(0, lambda: messagebox.showerror("Error", f"Could not open video file: {self.input_path}"))
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec based on output file extension
        # Common codecs: 'mp4v' for .mp4, 'XVID' for .avi
        # Ensure the chosen codec is available on the system
        fourcc = 0
        if self.output_video_path.lower().endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif self.output_video_path.lower().endswith(".avi"):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
        all_detected_lps = [] # To store all unique LPs detected across the video

        try:
            while cap.isOpened() and not self.stop_processing:
                ret, frame = cap.read()
                if not ret: # End of video or error reading frame
                    break

                results = model(frame, imgsz=640, verbose=False)

                img_with_boxes_and_text = frame.copy()
                current_frame_lps = []

                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    if class_id == 0: # License plate
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                        plate_crop = frame[y1:y2, x1:x2]

                        if plate_crop.shape[0] == 0 or plate_crop.shape[1] == 0:
                            continue # Skip invalid crops

                        final_lp = getLPText(plate_crop, ocr)
                        if final_lp and final_lp != "No text detected":
                            current_frame_lps.append(final_lp)
                            # Add to overall list if not already present (for unique LPs)
                            if final_lp not in all_detected_lps:
                                all_detected_lps.append(final_lp)

                        cv2.rectangle(img_with_boxes_and_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8 # Slightly smaller font for video overlay
                        font_thickness = 2
                        text_color = (0, 255, 0)
                        (text_width, text_height), baseline = cv2.getTextSize(final_lp, font, font_scale, font_thickness)
                        text_x = x1
                        text_y = y1 - 5 if y1 - 5 > text_height else y1 + text_height + 5 # Position text above box
                        cv2.putText(img_with_boxes_and_text, final_lp, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                if out:
                    out.write(img_with_boxes_and_text)

                frame_count += 1
                progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0

                # Update GUI from the main thread (using master.after)
                self.master.after(0, lambda f=img_with_boxes_and_text, p=progress, lps=current_frame_lps:
                                  self._update_video_gui_elements(f, p, lps))

            if self.stop_processing:
                self.master.after(0, lambda: self.status_label.config(text="Status: Processing Stopped."))
            else:
                self.master.after(0, lambda: self.status_label.config(text="Status: Video Processing Complete."))
                # Optionally save all detected LPs to a text file after completion
                if all_detected_lps:
                    self.master.after(0, lambda: self._prompt_save_detected_lps(all_detected_lps))


        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred during video processing: {e}"))
        finally:
            cap.release()
            if out:
                out.release()
            # The _check_processing_thread will handle the final state reset


    def _update_video_gui_elements(self, frame, progress, current_frame_lps):
        """Updates the image display, progress bar, and text label during video processing."""
        self._display_image_from_array(frame)
        self.progress_bar['value'] = progress
        self.result_label.config(text="Last Frame LPs: " + (", ".join(current_frame_lps) if current_frame_lps else "N/A"))
        self.master.update_idletasks() # Force GUI update to ensure smooth progress bar/frame display

    def _prompt_save_detected_lps(self, lps_list):
        """Asks the user if they want to save all detected LPs to a file."""
        if messagebox.askyesno("Save Detected LPs", "Do you want to save all detected license plates to a text file?"):
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save All Detected License Plates As"
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        for lp in lps_list:
                            f.write(lp + '\n')
                    messagebox.showinfo("Save Complete", f"All detected license plates saved to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Could not save file: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LPRecognitionApp(root)
    root.mainloop()