import ultralytics 
from ultralytics import YOLO
import cv2
# Load a pre-trained YOLO model
model = YOLO("lp_detect.pt") # Ensure this file is in the correct path or provide full path

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with detections
    annotated_frame = results[0].plot() # Ultralytics provides a convenient plot method

    # Display the annotated frame
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()