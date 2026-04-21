import cv2

# Test camera access
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully.")
    else:
        print("Error: Could not capture frame.")
    cap.release()