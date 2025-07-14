import cv2
import numpy as np
import mediapipe as mp
import time
import os
import threading
import queue
import math
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe cargado correctamente")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"❌ Error importando MediaPipe: {e}")
    print("🔄 Instalar con: pip install mediapipe")

class MediaPipeEmotionRecognition:
    def __init__(self, save_data=True):
        self.save_data = save_data
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 128, 0),
            'fear': (128, 0, 128),
            'happy': (0, 255, 255),
            'sad': (255, 0, 0),
            'surprise': (0, 165, 255),
            'neutral': (128, 128, 128)
        }
        self.face_trackers = {}
        self.next_face_id = 0
        self.emotion_buffers = {}
        self.buffer_size = 8
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.running = True
        self.debug_mode = True

        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("🎭 MediaPipe Face Mesh inicializado")
        else:
            print("⚠️  MediaPipe no disponible, usando modo mock")

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_landmark_points(self, landmarks, image_width, image_height):
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            points.append((x, y))
        return points

    def analyze_facial_features(self, landmarks, image_width, image_height):
        points = self.get_landmark_points(landmarks, image_width, image_height)
        features = {}
        try:
            mouth_left = points[61]
            mouth_right = points[291]
            mouth_top = points[13]
            mouth_bottom = points[14]
            mouth_width = self.calculate_distance(mouth_left, mouth_right)
            mouth_height = self.calculate_distance(mouth_top, mouth_bottom)
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            mouth_center = ((mouth_left[0] + mouth_right[0]) // 2, (mouth_left[1] + mouth_right[1]) // 2)
            mouth_curvature = (mouth_center[1] - mouth_left[1] - mouth_right[1]) / 2

            left_eye_left = points[33]
            left_eye_right = points[133]
            left_eye_top = points[159]
            left_eye_bottom = points[145]
            right_eye_left = points[362]
            right_eye_right = points[263]
            right_eye_top = points[386]
            right_eye_bottom = points[374]
            left_eye_width = self.calculate_distance(left_eye_left, left_eye_right)
            left_eye_height = self.calculate_distance(left_eye_top, left_eye_bottom)
            left_ear = left_eye_height / left_eye_width if left_eye_width > 0 else 0
            right_eye_width = self.calculate_distance(right_eye_left, right_eye_right)
            right_eye_height = self.calculate_distance(right_eye_top, right_eye_bottom)
            right_ear = right_eye_height / right_eye_width if right_eye_width > 0 else 0
            avg_ear = (left_ear + right_ear) / 2

            left_eyebrow_center = points[70]
            right_eyebrow_center = points[300]
            left_eyebrow_height = left_eye_top[1] - left_eyebrow_center[1]
            right_eyebrow_height = right_eye_top[1] - right_eyebrow_center[1]
            avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
            face_height = abs(points[10][1] - points[152][1])
            normalized_eyebrow_height = avg_eyebrow_height / face_height if face_height > 0 else 0

            left_cheek = points[116]
            right_cheek = points[345]
            nose_tip = points[1]
            cheek_elevation = (nose_tip[1] - left_cheek[1] + nose_tip[1] - right_cheek[1]) / 2
            normalized_cheek_elevation = cheek_elevation / face_height if face_height > 0 else 0

            nose_left = points[235]
            nose_right = points[455]
            nose_width = self.calculate_distance(nose_left, nose_right)
            nose_wrinkle_ratio = nose_width / face_height if face_height > 0 else 0

            features = {
                'mouth_aspect_ratio': mouth_aspect_ratio,
                'mouth_curvature': mouth_curvature,
                'mouth_width': mouth_width,
                'eye_aspect_ratio': avg_ear,
                'eyebrow_height': normalized_eyebrow_height,
                'cheek_elevation': normalized_cheek_elevation,
                'nose_wrinkle': nose_wrinkle_ratio,
                'face_height': face_height
            }
            return features
        except Exception as e:
            print(f"Error en análisis facial: {e}")
            return {
                'mouth_aspect_ratio': 0.5,
                'mouth_curvature': 0,
                'mouth_width': 50,
                'eye_aspect_ratio': 0.3,
                'eyebrow_height': 0.5,
                'cheek_elevation': 0.5,
                'nose_wrinkle': 0.5,
                'face_height': 100
            }

    def features_to_emotion(self, features):
        try:
            mar = features['mouth_aspect_ratio']
            ear = features['eye_aspect_ratio']
            brow_height = features['eyebrow_height']
            cheek_elev = features['cheek_elevation']
            mouth_curv = features['mouth_curvature']
            nose_wrinkle = features['nose_wrinkle']
            emotions = {}
            happiness_score = 0
            if mar > 0.02:
                happiness_score += 0.3
            if mouth_curv > 0:
                happiness_score += 0.4
            if cheek_elev > 0.3:
                happiness_score += 0.3
            if 0.15 < ear < 0.3:
                happiness_score += 0.2
            emotions['happy'] = max(0, min(1, happiness_score))
            sadness_score = 0
            if mouth_curv < -0.01:
                sadness_score += 0.3
            if brow_height < 0.3:
                sadness_score += 0.3
            if ear < 0.2:
                sadness_score += 0.2
            if mar < 0.01:
                sadness_score += 0.1
            emotions['sad'] = max(0, min(1, sadness_score))
            anger_score = 0
            if brow_height < 0.25:
                anger_score += 0.4
            if 0.1 < ear < 0.25:
                anger_score += 0.3
            if abs(mouth_curv) < 0.005:
                anger_score += 0.2
            if nose_wrinkle > 0.6:
                anger_score += 0.1
            emotions['angry'] = max(0, min(1, anger_score))
            surprise_score = 0
            if brow_height > 0.7:
                surprise_score += 0.4
            if ear > 0.4:
                surprise_score += 0.4
            if mar > 0.05:
                surprise_score += 0.3
            emotions['surprise'] = max(0, min(1, surprise_score))
            fear_score = 0
            if brow_height > 0.6:
                fear_score += 0.3
            if ear > 0.35:
                fear_score += 0.4
            if 0.02 < mar < 0.04:
                fear_score += 0.2
            if cheek_elev < 0.3:
                fear_score += 0.1
            emotions['fear'] = max(0, min(1, fear_score))
            disgust_score = 0
            if nose_wrinkle > 0.65:
                disgust_score += 0.4
            if mouth_curv < -0.005:
                disgust_score += 0.3
            if brow_height < 0.4:
                disgust_score += 0.2
            if cheek_elev < 0.2:
                disgust_score += 0.1
            emotions['disgust'] = max(0, min(1, disgust_score))
            max_emotion_score = max(emotions.values()) if emotions else 0
            neutral_score = max(0.2, 1 - max_emotion_score * 1.5)
            emotions['neutral'] = neutral_score
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            else:
                emotions = {emotion: 1/len(self.emotions) for emotion in self.emotions}
            emotions = self.smooth_emotions(emotions)
            return emotions
        except Exception as e:
            print(f"Error en clasificación de emociones: {e}")
            return {emotion: 1/len(self.emotions) for emotion in self.emotions}

    def smooth_emotions(self, emotions):
        smoothed = {}
        for emotion, value in emotions.items():
            smoothed_value = 1 / (1 + math.exp(-10 * (value - 0.5)))
            smoothed[emotion] = smoothed_value
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v/total for k, v in smoothed.items()}
        return smoothed

    def detect_emotions_mediapipe(self, frame):
        if not MEDIAPIPE_AVAILABLE:
            return self.mock_detection(frame)
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            detections = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w = frame.shape[:2]
                    x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                    y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    padding = 30
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    features = self.analyze_facial_features(face_landmarks, w, h)
                    emotions = self.features_to_emotion(features)
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    quality_score = self.calculate_quality_score(features, confidence)
                    detections.append({
                        "bbox": bbox,
                        "emotions": emotions,
                        "dominant_emotion": dominant_emotion,
                        "confidence": confidence,
                        "quality_score": quality_score,
                        "features": features,
                        "landmarks": face_landmarks,
                        "timestamp": time.time()
                    })
            return detections
        except Exception as e:
            print(f"Error en detección MediaPipe: {e}")
            return []

    def mock_detection(self, frame):
        h, w = frame.shape[:2]
        mock_detection = {
            "bbox": (w//4, h//4, w//2, h//2),
            "emotions": {
                'happy': 0.6,
                'neutral': 0.2,
                'sad': 0.1,
                'angry': 0.05,
                'surprise': 0.03,
                'fear': 0.01,
                'disgust': 0.01
            },
            "dominant_emotion": "happy",
            "confidence": 0.6,
            "quality_score": 0.8,
            "features": {
                'mouth_aspect_ratio': 0.5,
                'mouth_curvature': 0.1,
                'mouth_width': 50,
                'eye_aspect_ratio': 0.3,
                'eyebrow_height': 0.5,
                'cheek_elevation': 0.5,
                'nose_wrinkle': 0.5,
                'face_height': 100
            },
            "landmarks": None,
            "timestamp": time.time()
        }
        return [mock_detection]

    def calculate_quality_score(self, features, confidence):
        try:
            feature_values = [v for v in features.values() if isinstance(v, (int, float))]
            if not feature_values:
                return 0.3
            extreme_count = sum(1 for v in feature_values if v < 0.05 or v > 0.95)
            extreme_penalty = extreme_count / len(feature_values)
            invalid_count = sum(1 for v in feature_values if math.isnan(v) or math.isinf(v))
            invalid_penalty = invalid_count / len(feature_values)
            quality_score = confidence * (1 - extreme_penalty * 0.3 - invalid_penalty * 0.5)
            return max(0.1, min(1.0, quality_score))
        except Exception as e:
            print(f"Error calculando quality score: {e}")
            return 0.3

    def track_faces(self, detections):
        tracked_detections = []
        for detection in detections:
            bbox = detection["bbox"]
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            best_match_id = None
            min_distance = float('inf')
            for face_id, tracker_data in self.face_trackers.items():
                if time.time() - tracker_data["last_seen"] > 2.0:
                    continue
                tracker_center = tracker_data["center"]
                distance = np.sqrt(
                    (center_x - tracker_center[0])**2 +
                    (center_y - tracker_center[1])**2
                )
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    best_match_id = face_id
            if best_match_id is not None:
                face_id = best_match_id
            else:
                face_id = self.next_face_id
                self.next_face_id += 1
            self.face_trackers[face_id] = {
                "center": (center_x, center_y),
                "bbox": bbox,
                "last_seen": time.time(),
                "detection_count": self.face_trackers.get(face_id, {}).get("detection_count", 0) + 1
            }
            detection["face_id"] = face_id
            tracked_detections.append(detection)
        current_time = time.time()
        expired_trackers = [
            face_id for face_id, data in self.face_trackers.items()
            if current_time - data["last_seen"] > 3.0
        ]
        for face_id in expired_trackers:
            del self.face_trackers[face_id]
        return tracked_detections

    def stabilize_emotions(self, detections):
        stabilized_detections = []
        for detection in detections:
            face_id = detection.get("face_id")
            if face_id is None:
                stabilized_detections.append(detection)
                continue
            if face_id not in self.emotion_buffers:
                self.emotion_buffers[face_id] = deque(maxlen=self.buffer_size)
            self.emotion_buffers[face_id].append(detection)
            if len(self.emotion_buffers[face_id]) < 3:
                stabilized_detections.append(detection)
                continue
            buffer_data = list(self.emotion_buffers[face_id])
            weights = [0.5, 0.3, 0.2] if len(buffer_data) >= 3 else [1.0]
            if len(buffer_data) > 3:
                weights = [0.4, 0.3, 0.2, 0.1] + [0.0] * (len(buffer_data) - 4)
            weighted_emotions = {}
            for emotion in self.emotions:
                weighted_sum = 0
                weight_sum = 0
                for i, data in enumerate(buffer_data[-len(weights):]):
                    weight = weights[i] if i < len(weights) else 0
                    weighted_sum += data["emotions"].get(emotion, 0) * weight
                    weight_sum += weight
                weighted_emotions[emotion] = weighted_sum / weight_sum if weight_sum > 0 else 0
            stabilized_detection = detection.copy()
            stabilized_detection["emotions"] = weighted_emotions
            stabilized_detection["dominant_emotion"] = max(weighted_emotions, key=weighted_emotions.get)
            stabilized_detection["confidence"] = weighted_emotions[stabilized_detection["dominant_emotion"]]
            stabilized_detection["is_stabilized"] = True
            stabilized_detections.append(stabilized_detection)
        return stabilized_detections

    def emotion_worker(self):
        while self.running:
            try:
                if not self.frame_queue.empty():
                    with self.processing_lock:
                        if self.is_processing:
                            time.sleep(0.02)
                            continue
                        self.is_processing = True
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                        detections = self.detect_emotions_mediapipe(frame)
                        tracked_detections = self.track_faces(detections)
                        stabilized_detections = self.stabilize_emotions(tracked_detections)
                        if not self.result_queue.full():
                            self.result_queue.put(stabilized_detections)
                        if self.debug_mode and stabilized_detections:
                            print(f"📊 Procesadas {len(stabilized_detections)} caras")
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print(f"Error en worker: {e}")
                    finally:
                        with self.processing_lock:
                            self.is_processing = False
            except Exception as e:
                print(f"Error en emotion_worker loop: {e}")
                time.sleep(0.05)

    def stop(self):
        self.running = False
        print("🛑 Worker detenido")

if __name__ == "__main__":
    emotion_recognizer = MediaPipeEmotionRecognition(save_data=False)
    worker_thread = threading.Thread(target=emotion_recognizer.emotion_worker, daemon=True)
    worker_thread.start()
    cap = cv2.VideoCapture(0)
    print("🎥 Cámara iniciada. Presiona 'q' para salir.")
    try:
        while cap.isOpened() and emotion_recognizer.running:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No se pudo leer el frame de la cámara.")
                break
            if not emotion_recognizer.frame_queue.full():
                emotion_recognizer.frame_queue.put(frame)
            if not emotion_recognizer.result_queue.empty():
                detections = emotion_recognizer.result_queue.get()
                for det in detections:
                    x, y, w, h = det["bbox"]
                    color = emotion_recognizer.emotion_colors.get(det["dominant_emotion"], (255, 255, 255))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f'{det["dominant_emotion"]}: {det["confidence"]:.2f}'
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("MediaPipe Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("⏹ Interrupción por teclado.")
    finally:
        emotion_recognizer.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Programa finalizado correctamente.")