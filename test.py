from insightface.app import FaceAnalysis
import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FaceAnalysis with GPU support
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # Adjust detection size

# Load known faces
known_faces_dir = "known_faces"
known_embeddings = []
known_names = []

print("[INFO] Loading known faces...")
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(known_faces_dir, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to read {filename}")
            continue

        image = cv2.resize(image, (640, 640))
        faces = app.get(image)

        if faces:
            face = faces[0]
            known_embeddings.append(face.embedding)
            known_names.append(os.path.splitext(filename)[0])
            print(f"Loaded: {filename}")
        else:
            print(f"No face found in {filename}")

# Similarity threshold
threshold = 0.5

# Load the group image
input_path = "Screenshot 2025-06-20 194857.png"  # Replace with your input image path
frame = cv2.imread(input_path)

if frame is None:
    print("Error: Unable to read input image.")
    exit()

# Detect faces in the image
resized_frame = cv2.resize(frame, (640, 640))
detected_faces = app.get(resized_frame)

scale_x = frame.shape[1] / 640
scale_y = frame.shape[0] / 640

for face in detected_faces:
    emb = face.embedding.reshape(1, -1)
    similarities = cosine_similarity(emb, known_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    name = known_names[best_match_idx] if best_score > threshold else "Unknown"

    # Get bounding box and rescale to original image size
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # Draw rectangle and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow("Group Face Recognition", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the labeled image
cv2.imwrite("labeled_output.jpg", frame)
