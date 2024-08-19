import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Loop through the webcam frames
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if no frame is received
        break

# Release the webcam capture object and close the display window
cap.release()
cv2.destroyAllWindows()