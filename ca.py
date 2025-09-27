from eyetrax import GazeEstimator, run_9_point_calibration
import cv2
import pyautogui

from collections import deque

# Create estimator and calibrate
estimator = GazeEstimator()
run_9_point_calibration(estimator)

# # Save model
# estimator.save_model("gaze_model.pkl")

# # Load model
# estimator = GazeEstimator()
# estimator.load_model("gaze_model.pkl")
size = pyautogui.size()
cap = cv2.VideoCapture(0)
print("recording")

positions = deque(maxlen=10)

def check_sus():
    if len(positions) == 10:
        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        threshold = 100  # pixels
        similar = (max(xs) - min(xs) < threshold) and (max(ys) - min(ys) < threshold)
        out_of_bounds = all((x < 0 or x > size.width or y < 0 or y > size.height) for x, y in positions)

        if similar and out_of_bounds:
            print("Warning")


while True:
    # Extract features from frame
    ret, frame = cap.read()
    features, blink = estimator.extract_features(frame)

    # Predict screen coordinates
    if features is not None and not blink:
        x, y = estimator.predict([features])[0]

        positions.append((x, y))
        check_sus()
