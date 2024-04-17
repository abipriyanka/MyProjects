import cv2
import os
import time

root_dir = "gesture_data"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

train_dir = os.path.join(root_dir, "train")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

valid_dir = os.path.join(root_dir, "valid")
if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)

gestures = ["play", "pause", "next", "previous","volume_up","volume_down"]
for gesture in gestures:
    train_gesture_dir = os.path.join(train_dir, gesture)
    if not os.path.exists(train_gesture_dir):
        os.mkdir(train_gesture_dir)

    valid_gesture_dir = os.path.join(valid_dir, gesture)
    if not os.path.exists(valid_gesture_dir):
        os.mkdir(valid_gesture_dir)

cap = cv2.VideoCapture(0)
gesture = input("Enter gesture name: ")
is_train = input("Enter 't' for train directory and 'v' for valid directory: ").lower() == 't'
gesture_dir = os.path.join(train_dir if is_train else valid_dir, gesture)
if not os.path.exists(gesture_dir):
    os.mkdir(gesture_dir)

count = 0
start_time = None
while count < 150:
    ret, frame = cap.read()
    # convert frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    
    if start_time is None:
        cv2.putText(frame, f"Press 'k' to start capturing images", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        seconds_left = 1 - int(time.time() - start_time)
        if seconds_left < 0:
            seconds_left = 0
        cv2.putText(frame, f"Captured {count} images, {seconds_left}s left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Region of Interest", roi)
   
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('k'):
        start_time = time.time()
   
    if start_time is not None and time.time() - start_time >= 1:
        filename = os.path.join(gesture_dir, f"{gesture}_{count}.jpg")
        cv2.imwrite(filename, roi)
        count += 1
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()