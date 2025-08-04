import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
# Assuming this is your model class
from Model import TrafficSignNet

# Load model
model = TrafficSignNet()
model.load_state_dict(torch.load('/Users/arjuntomar/Desktop/PROJECT/gtsrb_model2.pth', map_location='cpu'))
model.eval()

# Define preprocessing (same as during training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # adjust if needed
])

# Mapping class index to name
class_names = [
    "Speed limit (20km/h)",                 # 0
    "Speed limit (30km/h)",                 # 1
    "Speed limit (50km/h)",                 # 2
    "Speed limit (60km/h)",                 # 3
    "Speed limit (70km/h)",                 # 4
    "Speed limit (80km/h)",                 # 5
    "End of speed limit (80km/h)",          # 6
    "Speed limit (100km/h)",                # 7
    "Speed limit (120km/h)",                # 8
    "No passing",                           # 9
    "No passing for vehicles over 3.5 metric tons", # 10
    "Right-of-way at the next intersection",# 11
    "Priority road",                        # 12
    "Yield",                                # 13
    "Stop",                                 # 14
    "No vehicles",                          # 15
    "Vehicles over 3.5 metric tons prohibited", # 16
    "No entry",                             # 17
    "General caution",                      # 18
    "Dangerous curve to the left",          # 19
    "Dangerous curve to the right",         # 20
    "Double curve",                         # 21
    "Bumpy road",                           # 22
    "Slippery road",                        # 23
    "Road narrows on the right",            # 24
    "Road work",                            # 25
    "Traffic signals",                      # 26
    "Pedestrians",                          # 27
    "Children crossing",                    # 28
    "Bicycles crossing",                    # 29
    "Beware of ice/snow",                   # 30
    "Wild animals crossing",                # 31
    "End of all speed and passing limits",  # 32
    "Turn right ahead",                     # 33
    "Turn left ahead",                      # 34
    "Ahead only",                           # 35
    "Go straight or right",                 # 36
    "Go straight or left",                  # 37
    "Keep right",                           # 38
    "Keep left",                            # 39
    "Roundabout mandatory",                 # 40
    "End of no passing",                    # 41
    "End of no passing by vehicles over 3.5 metric tons" # 42
]

# Improved sign detection with color ranges for traffic signs
def detect_signs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red range (for stop signs, etc.)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Blue range (for some information signs)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combined mask
    mask = red_mask | blue_mask
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Start webcam
cap = cv2.VideoCapture(0)

# Confidence threshold for showing detections
CONFIDENCE_THRESHOLD = 0.85

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contours = detect_signs(frame)
    
    boxes = []
    confidences = []
    labels = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filter by size - adjust these thresholds based on your setup
        if area < 1000 :  # Skip very small or very large detections
            continue
            
        # Check aspect ratio to filter out non-sign-like objects
        aspect_ratio = float(w) / h
        if aspect_ratio > 1.5 or aspect_ratio < 0.5:
            continue
            
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:  # Skip empty ROIs
            continue
            
        try:
            img_pil = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            input_tensor = transform(Image.fromarray(img_pil)).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                pred = output.argmax(dim=1).item()
                confidence = probabilities[pred].item()  # Get the confidence score
                
                # Only consider high confidence predictions
                if confidence > CONFIDENCE_THRESHOLD:
                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    labels.append(class_names[pred])
        except Exception as e:
            print(f"Error processing ROI: {e}")
            continue
    
    # Skip the NMS completely and just show the highest confidence detection
    if boxes and confidences:
        best_idx = np.argmax(confidences)
        x, y, w, h = boxes[best_idx]
        label = f"{labels[best_idx]} ({confidences[best_idx]:.2f})"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()