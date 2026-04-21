import cv2
import os
import sys

def register_user(name):
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    user_path = os.path.join(dataset_path, name)
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return
    print("Capturing photos for", name)
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow("Registration - Press 'q' to quit", frame)
        img_name = f"{user_path}/{count}.jpg"
        cv2.imwrite(img_name, frame)
        count += 1
        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images for {name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_name = sys.argv[1]
    else:
        user_name = input("Enter name to register: ")
    register_user(user_name)
