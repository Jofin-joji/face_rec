from insightface.app import FaceAnalysis
import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FaceAnalysis with GPU support
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) 

# Load known faces
known_faces_dir = "known_faces"
known_embeddings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(known_faces_dir, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to read {filename}")
            continue

        resized_image = cv2.resize(image, (640, 640))
        faces = app.get(resized_image)

        if len(faces) > 0:
            face = faces[0]
            known_embeddings.append(face.embedding)
            name = os.path.splitext(filename)[0]
            known_names.append(name)
            print(f"Loaded: {name}")
        else:
            print(f"No face found in {filename}")

# Set similarity threshold
threshold = 0.5

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    original_h, original_w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (640, 640))
    detected_faces = app.get(resized_frame)

    scale_x = original_w / 640
    scale_y = original_h / 640

    for face in detected_faces:
        emb = face.embedding.reshape(1, -1)
        similarities = cosine_similarity(emb, known_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        name = known_names[best_match_idx] if best_score > threshold else "Unknown"

        # Adjust bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Real-time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
