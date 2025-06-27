import cv2
import os
import threading
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ========== CONFIGURATION ==========
CAMERA_SOURCES = {
    "classroom_1": 0,  # Laptop webcam
    "classroom_2": 1   # Iriun mobile webcam (try 1 or the streaming URL)
}

KNOWN_FACES_DIR = "known_faces"
SIMILARITY_THRESHOLD = 0.5
attendance_log = {}

# ========== LOAD KNOWN FACES ==========
print("[INFO] Loading known face embeddings...")
known_embeddings = []
known_names = []

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, (640, 640))
        faces = app.get(image)
        if faces:
            known_embeddings.append(faces[0].embedding)
            known_names.append(os.path.splitext(filename)[0])
print(f"[INFO] Loaded {len(known_names)} known faces.")

# ========== CAMERA MONITORING FUNCTION ==========
def monitor_camera(camera_name, camera_source):
    print(f"[INFO] Starting stream for {camera_name}...")
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open {camera_name}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Failed to capture frame from {camera_name}.")
            break

        original_h, original_w = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640))
        detected_faces = app.get(resized)

        scale_x = original_w / 640
        scale_y = original_h / 640

        for face in detected_faces:
            emb = face.embedding.reshape(1, -1)
            similarities = cosine_similarity(emb, known_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            name = known_names[best_idx] if best_score > SIMILARITY_THRESHOLD else "Unknown"

            # Mark attendance
            if name != "Unknown":
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                key = f"{camera_name}_{name}"
                if key not in attendance_log:
                    attendance_log[key] = now
                    print(f"[ATTENDANCE] {name} marked present in {camera_name} at {now}")

            # Draw bounding box and name
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
            x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(f"{camera_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"{camera_name}")

# ========== START MONITORING THREADS ==========
threads = []
for cam_name, cam_src in CAMERA_SOURCES.items():
    t = threading.Thread(target=monitor_camera, args=(cam_name, cam_src))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("\n[INFO] Attendance Summary:")
for k, v in attendance_log.items():
    print(f"{k.split('_')[1]} => {k.split('_')[0]} at {v}")
