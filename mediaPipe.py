# mediaPipe_improved.py
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import json
from datetime import datetime
from collections import Counter, deque
import threading
import queue
import math

# Intentar importar dependencias opcionales
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe cargado correctamente")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"❌ Error importando MediaPipe: {e}")
    print("🔄 Instalar con: pip install mediapipe")

try:
    from scipy import signal
    from sklearn.preprocessing import StandardScaler
    ADVANCED_FEATURES = True
    print("✅ Características avanzadas disponibles")
except ImportError:
    ADVANCED_FEATURES = False
    print("⚠️ scipy y sklearn no disponibles. Instalar con: pip install scipy scikit-learn")

class ImprovedMediaPipeEmotionRecognition:
    def __init__(self, save_data=True):
        """Inicializa el sistema mejorado de reconocimiento de emociones"""
        self.save_data = save_data
        self.data_path = "emotion_data_improved"
        
        # Configuraciones optimizadas
        self.process_every_n_frames = 1
        self.confidence_threshold = 0.6
        self.stabilization_frames = 10
        
        # Inicializar MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            print("🎭 MediaPipe Face Mesh inicializado (modo mejorado)")
        
        # Definiciones de emociones
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'angry': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (128, 0, 128),
            'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (0, 165, 255),
            'neutral': (128, 128, 128)
        }
        
        # Índices de landmarks mejorados
        self.improved_landmarks = {
            'mouth_outer': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'mouth_corners': [61, 291, 39, 181],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            'right_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
            'nose_bridge': [6, 168, 8, 9, 10, 151],
            'nose_tip': [1, 2, 5, 4, 19, 20],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142],
            'right_cheek': [345, 346, 347, 348, 349, 350, 451, 452],
            'jaw_line': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
        
        # Sistema de calibración
        self.calibration_data = {}
        self.calibration_frames = 30
        self.is_calibrating = False
        self.calibration_emotion = None
        self.calibration_samples = []
        
        # Sistema de tracking
        self.face_trackers = {}
        self.next_face_id = 0
        self.emotion_buffers = {}
        self.feature_buffers = {}
        self.buffer_size = 15
        
        # Filtros de señal
        if ADVANCED_FEATURES:
            self.setup_signal_filters()
        
        # Estadísticas
        self.emotion_history = []
        self.detection_count = 0
        self.start_time = time.time()
        self.fps_counter = deque(maxlen=60)
        self.quality_scores = deque(maxlen=100)
        
        # Control de procesamiento
        self.last_process_time = 0
        self.process_interval = 0.033
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing_lock = threading.Lock()
        self.is_processing = False
        
        # Configuración de visualización
        self.show_landmarks = True
        self.show_features = True
        self.show_confidence_bars = True
        self.running = True
        self.frame_counter = 0
        
        # Crear directorio
        if self.save_data and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        self.load_calibration_model()
    
    def setup_signal_filters(self):
        """Configura filtros de señal para estabilización"""
        if not ADVANCED_FEATURES:
            return
        
        self.emotion_filter_order = 3
        self.emotion_filter_cutoff = 0.1
        self.emotion_filters = {}
        
        for emotion in self.emotions:
            try:
                self.emotion_filters[emotion] = signal.butter(
                    self.emotion_filter_order, 
                    self.emotion_filter_cutoff, 
                    btype='low'
                )
            except:
                pass
    
    def calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_advanced_features(self, landmarks, image_width, image_height):
        """Calcula características faciales avanzadas"""
        features = {}
        
        # Convertir landmarks a coordenadas
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            landmark_points.append((x, y))
        
        # Análisis de boca
        mouth_features = self.analyze_mouth_advanced(landmark_points)
        features.update(mouth_features)
        
        # Análisis de ojos
        eye_features = self.analyze_eyes_advanced(landmark_points)
        features.update(eye_features)
        
        # Análisis de cejas
        eyebrow_features = self.analyze_eyebrows_advanced(landmark_points)
        features.update(eyebrow_features)
        
        # Análisis de mejillas
        cheek_features = self.analyze_cheeks_advanced(landmark_points)
        features.update(cheek_features)
        
        # Análisis de simetría
        symmetry_features = self.analyze_facial_symmetry(landmark_points)
        features.update(symmetry_features)
        
        return features
    
    def analyze_mouth_advanced(self, landmark_points):
        """Análisis avanzado de la boca"""
        try:
            mouth_outer = [landmark_points[i] for i in self.improved_landmarks['mouth_outer']]
            mouth_corners = [landmark_points[i] for i in self.improved_landmarks['mouth_corners']]
            
            # Curvatura de la boca
            left_corner = mouth_corners[0]
            right_corner = mouth_corners[1]
            mouth_center = mouth_outer[6] if len(mouth_outer) > 6 else mouth_outer[0]
            
            curvature = self.calculate_curvature_3points(left_corner, mouth_center, right_corner)
            
            # Apertura de la boca
            mouth_top = mouth_outer[3] if len(mouth_outer) > 3 else mouth_outer[0]
            mouth_bottom = mouth_outer[9] if len(mouth_outer) > 9 else mouth_outer[0]
            mouth_height = self.calculate_distance(mouth_top, mouth_bottom)
            mouth_width = self.calculate_distance(left_corner, right_corner)
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
            
            return {
                'mouth_curvature': curvature,
                'mouth_aspect_ratio': mouth_aspect_ratio,
                'mouth_width': mouth_width,
                'mouth_height': mouth_height
            }
        except:
            return {
                'mouth_curvature': 0.0,
                'mouth_aspect_ratio': 0.3,
                'mouth_width': 50.0,
                'mouth_height': 15.0
            }
    
    def analyze_eyes_advanced(self, landmark_points):
        """Análisis avanzado de los ojos"""
        try:
            left_eye = [landmark_points[i] for i in self.improved_landmarks['left_eye'][:6]]
            right_eye = [landmark_points[i] for i in self.improved_landmarks['right_eye'][:6]]
            
            left_eye_ratio = self.calculate_eye_aspect_ratio(left_eye)
            right_eye_ratio = self.calculate_eye_aspect_ratio(right_eye)
            avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
            eye_asymmetry = abs(left_eye_ratio - right_eye_ratio)
            
            return {
                'eye_aspect_ratio': avg_eye_ratio,
                'eye_asymmetry': eye_asymmetry,
                'left_eye_ratio': left_eye_ratio,
                'right_eye_ratio': right_eye_ratio
            }
        except:
            return {
                'eye_aspect_ratio': 0.3,
                'eye_asymmetry': 0.1,
                'left_eye_ratio': 0.3,
                'right_eye_ratio': 0.3
            }
    
    def analyze_eyebrows_advanced(self, landmark_points):
        """Análisis avanzado de las cejas"""
        try:
            left_eyebrow = [landmark_points[i] for i in self.improved_landmarks['left_eyebrow'][:5]]
            right_eyebrow = [landmark_points[i] for i in self.improved_landmarks['right_eyebrow'][:5]]
            
            left_eye_center = landmark_points[33]
            right_eye_center = landmark_points[362]
            
            left_height = self.calculate_eyebrow_height(left_eyebrow, left_eye_center)
            right_height = self.calculate_eyebrow_height(right_eyebrow, right_eye_center)
            avg_height = (left_height + right_height) / 2
            
            return {
                'eyebrow_height': avg_height,
                'eyebrow_asymmetry': abs(left_height - right_height)
            }
        except:
            return {
                'eyebrow_height': 0.5,
                'eyebrow_asymmetry': 0.1
            }
    
    def analyze_cheeks_advanced(self, landmark_points):
        """Análisis avanzado de mejillas"""
        try:
            left_cheek = [landmark_points[i] for i in self.improved_landmarks['left_cheek']]
            right_cheek = [landmark_points[i] for i in self.improved_landmarks['right_cheek']]
            
            left_raise = self.calculate_cheek_raise(left_cheek)
            right_raise = self.calculate_cheek_raise(right_cheek)
            avg_raise = (left_raise + right_raise) / 2
            
            return {
                'cheek_raise': avg_raise,
                'cheek_asymmetry': abs(left_raise - right_raise)
            }
        except:
            return {
                'cheek_raise': 0.5,
                'cheek_asymmetry': 0.1
            }
    
    def analyze_facial_symmetry(self, landmark_points):
        """Análisis de simetría facial"""
        try:
            nose_tip = landmark_points[1]
            left_points = [landmark_points[33], landmark_points[61]]  # Ojo y boca izquierda
            right_points = [landmark_points[362], landmark_points[291]]  # Ojo y boca derecha
            
            asymmetry = 0
            for left_p, right_p in zip(left_points, right_points):
                left_dist = abs(left_p[0] - nose_tip[0])
                right_dist = abs(right_p[0] - nose_tip[0])
                asymmetry += abs(left_dist - right_dist)
            
            symmetry = max(0, 1.0 - (asymmetry / 100))
            
            return {'facial_symmetry': symmetry}
        except:
            return {'facial_symmetry': 0.8}
    
    def calculate_curvature_3points(self, p1, p2, p3):
        """Calcula curvatura usando 3 puntos"""
        try:
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                return cross_product / (mag1 * mag2)
        except:
            pass
        return 0
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calcula ratio de aspecto del ojo"""
        if len(eye_points) < 6:
            return 0.3
        
        try:
            vertical_1 = self.calculate_distance(eye_points[1], eye_points[5])
            vertical_2 = self.calculate_distance(eye_points[2], eye_points[4])
            horizontal = self.calculate_distance(eye_points[0], eye_points[3])
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return max(0.1, min(1.0, ear))
        except:
            return 0.3
    
    def calculate_eyebrow_height(self, eyebrow_points, eye_center):
        """Calcula altura de ceja"""
        if len(eyebrow_points) < 3:
            return 0.5
        
        try:
            avg_y = sum(point[1] for point in eyebrow_points) / len(eyebrow_points)
            distance = abs(avg_y - eye_center[1])
            normalized_height = min(1.0, distance / 50)
            return normalized_height
        except:
            return 0.5
    
    def calculate_cheek_raise(self, cheek_points):
        """Calcula elevación de mejilla"""
        if len(cheek_points) < 3:
            return 0.5
        
        try:
            upper_y = sum(point[1] for point in cheek_points[:2]) / 2
            lower_y = sum(point[1] for point in cheek_points[-2:]) / 2
            elevation = max(0, min(1, (lower_y - upper_y) / 50))
            return elevation
        except:
            return 0.5
    
    def improved_emotion_recognition(self, features):
        """Sistema mejorado de reconocimiento de emociones"""
        emotions = {}
        
        # Extraer características
        mouth_curve = features.get('mouth_curvature', 0.0)
        mouth_ratio = features.get('mouth_aspect_ratio', 0.3)
        eye_ratio = features.get('eye_aspect_ratio', 0.3)
        eyebrow_height = features.get('eyebrow_height', 0.5)
        cheek_raise = features.get('cheek_raise', 0.5)
        
        # FELICIDAD - sonrisa, mejillas elevadas
        happiness = 0
        if mouth_curve > 0:
            happiness += mouth_curve * 0.4
        happiness += max(0, cheek_raise - 0.5) * 0.3
        happiness += max(0, 0.7 - eye_ratio) * 0.2
        happiness += 0.1
        
        # TRISTEZA - boca hacia abajo, cejas bajas
        sadness = 0
        if mouth_curve < 0:
            sadness += abs(mouth_curve) * 0.3
        sadness += max(0, 0.4 - eyebrow_height) * 0.3
        sadness += max(0, 0.6 - eye_ratio) * 0.2
        sadness += max(0, 0.4 - cheek_raise) * 0.2
        
        # ENOJO - cejas fruncidas, boca tensa
        anger = 0
        anger += max(0, 0.3 - eyebrow_height) * 0.4
        anger += max(0, eye_ratio - 0.6) * 0.3
        anger += max(0, 0.4 - mouth_ratio) * 0.3
        
        # SORPRESA - cejas altas, ojos abiertos, boca abierta
        surprise = 0
        surprise += max(0, eyebrow_height - 0.7) * 0.4
        surprise += max(0, eye_ratio - 0.5) * 0.3
        surprise += max(0, mouth_ratio - 0.3) * 0.3
        
        # MIEDO - cejas altas, ojos muy abiertos
        fear = 0
        fear += max(0, eyebrow_height - 0.6) * 0.4
        fear += max(0, eye_ratio - 0.6) * 0.4
        fear += max(0, mouth_ratio - 0.1) * 0.2
        
        # DISGUSTO - boca hacia abajo, cejas ligeramente fruncidas
        disgust = 0
        if mouth_curve < 0:
            disgust += abs(mouth_curve) * 0.4
        disgust += max(0, 0.4 - eyebrow_height) * 0.3
        disgust += max(0, 0.5 - cheek_raise) * 0.3
        
        # NEUTRAL - estado base
        neutral = 1.0 - max(happiness, sadness, anger, surprise, fear, disgust)
        neutral = max(0.1, neutral)
        
        emotions = {
            'happy': happiness,
            'sad': sadness,
            'angry': anger,
            'surprise': surprise,
            'fear': fear,
            'disgust': disgust,
            'neutral': neutral
        }
        
        # Normalizar
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        # Aplicar calibración
        emotions = self.apply_calibration(emotions)
        
        return emotions
    
    def apply_calibration(self, emotions):
        """Aplica calibración personalizada"""
        if not self.calibration_data:
            return emotions
        
        calibrated_emotions = {}
        for emotion, value in emotions.items():
            baseline = self.calibration_data.get(f'{emotion}_baseline', 0.0)
            scale = self.calibration_data.get(f'{emotion}_scale', 1.0)
            
            calibrated_value = (value - baseline) * scale
            calibrated_emotions[emotion] = max(0, min(1, calibrated_value))
        
        # Renormalizar
        total = sum(calibrated_emotions.values())
        if total > 0:
            calibrated_emotions = {k: v/total for k, v in calibrated_emotions.items()}
        
        return calibrated_emotions
    
    def start_calibration(self, emotion='neutral'):
        """Inicia calibración"""
        self.is_calibrating = True
        self.calibration_emotion = emotion
        self.calibration_samples = []
        print(f"🎯 Calibrando '{emotion}'. Mantén esa expresión por {self.calibration_frames} frames.")
    
    def process_calibration_frame(self, features):
        """Procesa frame durante calibración"""
        if not self.is_calibrating:
            return
        
        self.calibration_samples.append(features.copy())
        
        if len(self.calibration_samples) >= self.calibration_frames:
            self.finish_calibration()
    
    def finish_calibration(self):
        """Finaliza calibración"""
        if not self.calibration_samples:
            return
        
        emotion = self.calibration_emotion
        
        # Calcular emociones promedio
        emotion_samples = []
        for features in self.calibration_samples:
            emotions = self.improved_emotion_recognition(features)
            emotion_samples.append(emotions)
        
        # Calcular baseline para cada emoción
        for emo in self.emotions:
            values = [sample[emo] for sample in emotion_samples]
            baseline = sum(values) / len(values)
            self.calibration_data[f'{emo}_baseline'] = baseline
            
            if emo == emotion:
                target_value = 0.8
                scale = target_value / baseline if baseline > 0 else 1.0
            else:
                scale = 0.5
            
            self.calibration_data[f'{emo}_scale'] = scale
        
        self.is_calibrating = False
        self.save_calibration_model()
        print(f"✅ Calibración completada para '{emotion}'")
    
    def save_calibration_model(self):
        """Guarda modelo de calibración"""
        try:
            calibration_file = f"{self.data_path}/calibration_model.json"
            with open(calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
            print(f"💾 Calibración guardada")
        except Exception as e:
            print(f"❌ Error guardando calibración: {e}")
    
    def load_calibration_model(self):
        """Carga modelo de calibración"""
        try:
            calibration_file = f"{self.data_path}/calibration_model.json"
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                print(f"📂 Calibración cargada")
        except Exception as e:
            print(f"⚠️ No se pudo cargar calibración")
    
    def detect_emotions_improved(self, frame):
        """Detecta emociones usando el sistema mejorado"""
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
                    
                    # Características avanzadas
                    features = self.calculate_advanced_features(face_landmarks, w, h)
                    
                    # Procesar calibración
                    if self.is_calibrating:
                        self.process_calibration_frame(features)
                    
                    # Reconocer emociones
                    emotions = self.improved_emotion_recognition(features)
                    
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    
                    quality_score = self.calculate_quality_score(features, emotions)
                    
                    detections.append({
                        "bbox": bbox,
                        "emotions": emotions,
                        "dominant_emotion": dominant_emotion,
                        "confidence": confidence,
                        "quality_score": quality_score,
                        "features": features,
                        "landmarks": face_landmarks,
                        "timestamp": time.time(),
                        "frame_id": self.frame_counter
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error en detección: {e}")
            return []
    
    def calculate_quality_score(self, features, emotions):
        """Calcula score de calidad"""
        symmetry = features.get('facial_symmetry', 0.5)
        eye_consistency = 1.0 - features.get('eye_asymmetry', 0.5)
        
        sorted_emotions = sorted(emotions.values(), reverse=True)
        emotion_focus = sorted_emotions[0] - sorted_emotions[1] if len(sorted_emotions) > 1 else 1.0
        
        quality = (symmetry * 0.4 + eye_consistency * 0.3 + emotion_focus * 0.3)
        
        return max(0.2, min(1.0, quality))
    
    def advanced_tracking(self, detections):
        """Sistema de tracking avanzado"""
        tracked_detections = []
        current_time = time.time()
        
        for detection in detections:
            bbox = detection["bbox"]
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            current_center = (center_x, center_y)
            
            best_match_id = None
            min_distance = float('inf')
            
            # Limpiar trackers antiguos
            for face_id in list(self.face_trackers.keys()):
                if current_time - self.face_trackers[face_id]["last_seen"] > 2.0:
                    del self.face_trackers[face_id]
                    if face_id in self.emotion_buffers:
                        del self.emotion_buffers[face_id]
                    if face_id in self.feature_buffers:
                        del self.feature_buffers[face_id]
            
            # Buscar mejor match
            for face_id, tracker_data in self.face_trackers.items():
                tracker_center = tracker_data["center"]
                distance = math.sqrt(
                    (center_x - tracker_center[0])**2 + 
                    (center_y - tracker_center[1])**2
                )
                
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    best_match_id = face_id
            
            # Asignar ID
            if best_match_id is not None:
                face_id = best_match_id
            else:
                face_id = self.next_face_id
                self.next_face_id += 1
                self.emotion_buffers[face_id] = deque(maxlen=self.buffer_size)
                self.feature_buffers[face_id] = deque(maxlen=self.buffer_size)
            
            # Actualizar tracker
            self.face_trackers[face_id] = {
                "center": current_center,
                "bbox": bbox,
                "last_seen": current_time
            }
            
            detection["face_id"] = face_id
            tracked_detections.append(detection)
        
        return tracked_detections
    
    def advanced_stabilization(self, detections):
        """Estabilización avanzada"""
        stabilized_detections = []
        
        for detection in detections:
            face_id = detection.get("face_id")
            
            if face_id is None:
                stabilized_detections.append(detection)
                continue
            
            # Añadir al buffer
            self.emotion_buffers[face_id].append(detection["emotions"])
            self.feature_buffers[face_id].append(detection["features"])
            
            # Estabilizar si hay suficientes muestras
            if len(self.emotion_buffers[face_id]) < self.stabilization_frames:
                stabilized_detections.append(detection)
                continue
            
            # Aplicar estabilización
            stabilized_emotions = self.apply_stabilization(face_id)
            
            stabilized_detection = detection.copy()
            stabilized_detection["emotions"] = stabilized_emotions
            stabilized_detection["dominant_emotion"] = max(stabilized_emotions, key=stabilized_emotions.get)
            stabilized_detection["confidence"] = stabilized_emotions[stabilized_detection["dominant_emotion"]]
            stabilized_detection["is_stabilized"] = True
            
            stability_confidence = self.calculate_stability_confidence(face_id)
            stabilized_detection["stability_confidence"] = stability_confidence
            
            stabilized_detections.append(stabilized_detection)
        
        return stabilized_detections
    
    def apply_stabilization(self, face_id):
        """Aplica estabilización usando promedio ponderado"""
        emotion_buffer = list(self.emotion_buffers[face_id])
        
        if not emotion_buffer:
            return {emotion: 1/len(self.emotions) for emotion in self.emotions}
        
        # Pesos decrecientes (más peso a muestras recientes)
        weights = [i+1 for i in range(len(emotion_buffer))]
        total_weight = sum(weights)
        
        stabilized_emotions = {}
        for emotion in self.emotions:
            weighted_sum = sum(sample[emotion] * weight 
                             for sample, weight in zip(emotion_buffer, weights))
            stabilized_emotions[emotion] = weighted_sum / total_weight
        
        return stabilized_emotions
    
    def calculate_stability_confidence(self, face_id):
        """Calcula confianza de estabilización"""
        if face_id not in self.emotion_buffers:
            return 0.5
        
        emotion_buffer = list(self.emotion_buffers[face_id])
        if len(emotion_buffer) < 3:
            return 0.3
        
        # Calcular consistencia de emoción dominante
        recent_samples = emotion_buffer[-5:]
        dominant_emotions = [max(sample, key=sample.get) for sample in recent_samples]
        
        most_common = max(set(dominant_emotions), key=dominant_emotions.count)
        consistency = dominant_emotions.count(most_common) / len(dominant_emotions)
        
        sample_factor = min(1.0, len(emotion_buffer) / self.stabilization_frames)
        confidence = (consistency * 0.7 + sample_factor * 0.3)
        
        return confidence
    
    def draw_advanced_visualization(self, frame, results):
        """Visualización avanzada con más información"""
        for result in results:
            bbox = result["bbox"]
            x, y, w, h = bbox
            
            dominant_emotion = result["dominant_emotion"]
            confidence = result["confidence"]
            emotions = result["emotions"]
            face_id = result.get("face_id", "?")
            quality_score = result.get("quality_score", 0)
            is_stabilized = result.get("is_stabilized", False)
            stability_confidence = result.get("stability_confidence", 0)
            
            # Color dinámico
            base_color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            intensity = min(1.0, (confidence + quality_score) / 2)
            color = tuple(int(c * intensity) for c in base_color)
            
            # Grosor basado en calidad
            thickness = max(1, int(quality_score * 4))
            
            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Indicadores de estado
            if is_stabilized:
                cv2.circle(frame, (x + w - 15, y + 15), 6, (0, 255, 0), -1)
            
            if self.is_calibrating:
                cv2.circle(frame, (x + w - 30, y + 15), 8, (255, 255, 0), -1)
            
            # Landmarks si está habilitado
            if self.show_landmarks and "landmarks" in result and MEDIAPIPE_AVAILABLE:
                self.mp_drawing.draw_landmarks(
                    frame, result["landmarks"], 
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, 
                    self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # Panel de información
            panel_height = 100
            cv2.rectangle(frame, (x, y - panel_height), (x + w, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y - panel_height), (x + w, y), color, 2)
            
            # Textos informativos
            texts = [
                f"ID{face_id}: {dominant_emotion.capitalize()}",
                f"Conf: {confidence:.2f} | Qual: {quality_score:.2f}",
                f"Stab: {stability_confidence:.2f}"
            ]
            
            for i, text in enumerate(texts):
                cv2.putText(frame, text, (x + 5, y - panel_height + 20 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Barras de confianza
            if self.show_confidence_bars:
                self.draw_emotion_bars(frame, emotions, x, y - panel_height + 70, w)
            
            # Características
            if self.show_features and "features" in result:
                self.draw_feature_indicators(frame, result["features"], x + w + 10, y)
    
    def draw_emotion_bars(self, frame, emotions, start_x, start_y, width):
        """Dibuja barras de confianza para emociones"""
        bar_height = 6
        bar_spacing = 2
        
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, confidence) in enumerate(sorted_emotions[:4]):
            if confidence < 0.1:
                continue
                
            y_pos = start_y + i * (bar_height + bar_spacing)
            bar_width = int(width * confidence)
            color = self.emotion_colors.get(emotion, (128, 128, 128))
            
            # Barra de fondo
            cv2.rectangle(frame, (start_x, y_pos), (start_x + width, y_pos + bar_height), 
                         (50, 50, 50), -1)
            
            # Barra de confianza
            if bar_width > 0:
                cv2.rectangle(frame, (start_x, y_pos), (start_x + bar_width, y_pos + bar_height), 
                             color, -1)
            
            # Etiqueta
            cv2.putText(frame, f"{emotion[:3]}", (start_x + width + 5, y_pos + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def draw_feature_indicators(self, frame, features, start_x, start_y):
        """Dibuja indicadores de características"""
        indicator_size = 40
        indicators_per_row = 2
        
        key_features = [
            ('mouth_curvature', 'Mouth'),
            ('eye_aspect_ratio', 'Eyes'),
            ('eyebrow_height', 'Brows'),
            ('cheek_raise', 'Cheeks')
        ]
        
        for i, (feature_key, label) in enumerate(key_features):
            if feature_key not in features:
                continue
                
            row = i // indicators_per_row
            col = i % indicators_per_row
            
            x = start_x + col * (indicator_size + 10)
            y = start_y + row * (indicator_size + 15)
            
            value = features[feature_key]
            
            # Color basado en valor
            if value > 0.7:
                color = (0, 255, 0)
            elif value > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            # Círculo indicador
            radius = int(indicator_size//2 * min(1.0, abs(value)))
            cv2.circle(frame, (x + indicator_size//2, y + indicator_size//2), 
                      radius, color, -1)
            cv2.circle(frame, (x + indicator_size//2, y + indicator_size//2), 
                      indicator_size//2, (255, 255, 255), 1)
            
            # Etiqueta
            cv2.putText(frame, label, (x, y + indicator_size + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def draw_advanced_statistics(self, frame):
        """Dibuja estadísticas avanzadas"""
        fps = self.calculate_fps()
        
        # Calcular estadísticas
        avg_quality = sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0
        emotion_stats = Counter(self.emotion_history[-50:])
        most_common = emotion_stats.most_common(3)
        
        info_text = [
            f"MediaPipe Enhanced: {'Active' if MEDIAPIPE_AVAILABLE else 'Mock'} | FPS: {fps:.1f}",
            f"Tracked Faces: {len(self.face_trackers)} | Avg Quality: {avg_quality:.2f}",
            f"Detections: {self.detection_count} | Runtime: {time.time() - self.start_time:.1f}s",
            f"Calibration: {'Active' if self.is_calibrating else 'Ready' if self.calibration_data else 'None'}"
        ]
        
        # Estadísticas de emociones
        if most_common:
            emotion_text = " | ".join([f"{emo}: {count}" for emo, count in most_common])
            info_text.append(f"Recent: {emotion_text}")
        
        # Panel
        panel_height = len(info_text) * 22 + 20
        cv2.rectangle(frame, (10, 10), (480, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (480, 10 + panel_height), (100, 100, 100), 2)
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (15, 30 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Indicador de modo
        mode_text = f"Mode: {'Calibrating' if self.is_calibrating else 'Detecting'}"
        mode_color = (0, 255, 255) if self.is_calibrating else (0, 255, 0)
        cv2.putText(frame, mode_text, (15, 10 + panel_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    def emotion_worker_improved(self):
        """Worker thread mejorado"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    with self.processing_lock:
                        if self.is_processing:
                            time.sleep(0.01)
                            continue
                        self.is_processing = True
                    
                    try:
                        frame = self.frame_queue.get(timeout=0.05)
                        
                        # Detectar emociones
                        detections = self.detect_emotions_improved(frame)
                        
                        # Tracking
                        tracked_detections = self.advanced_tracking(detections)
                        
                        # Estabilización
                        stabilized_detections = self.advanced_stabilization(tracked_detections)
                        
                        # Actualizar estadísticas
                        for detection in stabilized_detections:
                            self.quality_scores.append(detection.get("quality_score", 0))
                        
                        # Enviar resultados
                        if not self.result_queue.full():
                            self.result_queue.put(stabilized_detections)
                    
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print(f"Error en worker: {e}")
                    finally:
                        with self.processing_lock:
                            self.is_processing = False
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error crítico en worker: {e}")
                time.sleep(0.5)
    
    def should_process_frame(self, frame_count):
        """Determina si procesar el frame"""
        current_time = time.time()
        
        if frame_count % self.process_every_n_frames != 0:
            return False
        
        if current_time - self.last_process_time < self.process_interval:
            return False
        
        self.last_process_time = current_time
        return True
    
    def handle_keyboard_input(self, key):
        """Maneja entrada de teclado"""
        if key == ord('q'):
            return False
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
            print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
        elif key == ord('f'):
            self.show_features = not self.show_features
            print(f"Features: {'ON' if self.show_features else 'OFF'}")
        elif key == ord('b'):
            self.show_confidence_bars = not self.show_confidence_bars
            print(f"Confidence bars: {'ON' if self.show_confidence_bars else 'OFF'}")
        elif key == ord('r'):
            self.reset_statistics()
        elif key == ord('s'):
            self.save_emotion_data_improved()
        elif key == ord('c'):
            if not self.is_calibrating:
                print("Ingresa emoción para calibrar:")
                print("1-Happy 2-Sad 3-Angry 4-Surprise 5-Fear 6-Disgust 7-Neutral")
        elif key >= ord('1') and key <= ord('7'):
            emotion_map = {
                ord('1'): 'happy', ord('2'): 'sad', ord('3'): 'angry',
                ord('4'): 'surprise', ord('5'): 'fear', ord('6'): 'disgust',
                ord('7'): 'neutral'
            }
            if not self.is_calibrating:
                self.start_calibration(emotion_map[key])
        
        return True
    
    def save_emotion_data_improved(self):
        """Guarda datos mejorados"""
        if not self.save_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_path}/emotions_improved_{timestamp}.json"
        
        # Estadísticas de calidad
        quality_stats = {
            "avg_quality": sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0,
            "min_quality": min(self.quality_scores) if self.quality_scores else 0,
            "max_quality": max(self.quality_scores) if self.quality_scores else 0
        }
        
        # Estadísticas de emociones
        emotion_counter = Counter(self.emotion_history)
        emotion_stats = {
            "total_detections": self.detection_count,
            "emotion_distribution": dict(emotion_counter),
            "most_common_emotion": emotion_counter.most_common(1)[0] if emotion_counter else None
        }
        
        data = {
            "timestamp": timestamp,
            "runtime": time.time() - self.start_time,
            "quality_stats": quality_stats,
            "emotion_stats": emotion_stats,
            "calibration_active": bool(self.calibration_data),
            "system_config": {
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "advanced_features": ADVANCED_FEATURES,
                "stabilization_frames": self.stabilization_frames,
                "buffer_size": self.buffer_size
            },
            "recent_emotions": self.emotion_history[-100:]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Datos guardados en {filename}")
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
    
    def calculate_fps(self):
        """Calcula FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                return (len(self.fps_counter) - 1) / time_diff
        return 0
    
    def reset_statistics(self):
        """Resetea estadísticas"""
        self.detection_count = 0
        self.start_time = time.time()
        self.emotion_history = []
        self.fps_counter.clear()
        self.quality_scores.clear()
        self.face_trackers.clear()
        self.emotion_buffers.clear()
        self.feature_buffers.clear()
        self.next_face_id = 0
        print("📊 Estadísticas reseteadas")
    
    def mock_detection(self, frame):
        """Detección mock mejorada"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            results = []
            for (x, y, w, h) in faces:
                # Simular emociones más realistas
                mock_emotions = {emotion: np.random.random() * 0.2 for emotion in self.emotions}
                
                # Hacer una emoción dominante
                dominant = np.random.choice(self.emotions)
                mock_emotions[dominant] = 0.4 + np.random.random() * 0.4
                
                # Normalizar
                total = sum(mock_emotions.values())
                mock_emotions = {k: v/total for k, v in mock_emotions.items()}
                
                # Mock features
                mock_features = {
                    'mouth_curvature': np.random.uniform(-0.5, 0.5),
                    'eye_aspect_ratio': np.random.uniform(0.2, 0.8),
                    'eyebrow_height': np.random.uniform(0.3, 0.8),
                    'facial_symmetry': np.random.uniform(0.7, 1.0),
                    'cheek_raise': np.random.uniform(0.2, 0.8)
                }
                
                results.append({
                    "bbox": (x, y, w, h),
                    "emotions": mock_emotions,
                    "dominant_emotion": dominant,
                    "confidence": mock_emotions[dominant],
                    "quality_score": np.random.uniform(0.5, 0.9),
                    "features": mock_features,
                    "timestamp": time.time()
                })
            
            return results
        except Exception as e:
            print(f"Error en mock detection: {e}")
            return []
    
    def cleanup_improved(self):
        """Limpieza mejorada de recursos"""
        self.running = False
        
        # Cerrar MediaPipe
        if MEDIAPIPE_AVAILABLE and hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        
        # Guardar datos finales
        self.save_emotion_data_improved()
        
        # Guardar calibración
        if self.calibration_data:
            self.save_calibration_model()
        
        print("🧹 Recursos liberados")
    
    def print_controls(self):
        """Imprime controles de teclado"""
        controls = """
🎮 CONTROLES MEJORADOS:
──────────────────────
q - Salir
l - Toggle landmarks
f - Toggle características  
b - Toggle barras de confianza
r - Reset estadísticas
s - Guardar datos
c - Ver opciones de calibración

CALIBRACIÓN RÁPIDA:
1 - Happy    2 - Sad      3 - Angry
4 - Surprise 5 - Fear     6 - Disgust  
7 - Neutral

💡 Para calibrar: Haz la expresión y presiona el número correspondiente
        """
        print(controls)
    
    def run_improved(self):
        """Ejecuta el sistema mejorado"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error al abrir la cámara")
            return
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Iniciar worker thread
        worker_thread = threading.Thread(target=self.emotion_worker_improved, daemon=True)
        worker_thread.start()
        
        print("🎥 Sistema mejorado iniciado")
        self.print_controls()
        
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ No se pudo leer el frame")
                    break
                
                frame_count += 1
                self.frame_counter = frame_count
                
                # Enviar frame a procesamiento
                if self.should_process_frame(frame_count):
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                
                # Mostrar resultados
                if not self.result_queue.empty():
                    detections = self.result_queue.get()
                    
                    # Actualizar estadísticas
                    self.detection_count += len(detections)
                    self.emotion_history.extend([d["dominant_emotion"] for d in detections])
                    
                    # Dibujar visualización
                    self.draw_advanced_visualization(frame, detections)
                
                # Dibujar estadísticas
                self.draw_advanced_statistics(frame)
                
                # Mostrar frame
                cv2.imshow("MediaPipe Emotion Recognition - IMPROVED", frame)
                
                # Manejar teclado
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup_improved()

# Función principal
if __name__ == "__main__":
    print("🚀 Iniciando MediaPipe Emotion Recognition - VERSIÓN MEJORADA")
    
    # Verificar dependencias
    print(f"📦 MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print(f"📦 Características avanzadas: {'✅' if ADVANCED_FEATURES else '❌'}")
    
    if not MEDIAPIPE_AVAILABLE:
        print("⚠️  Para instalar MediaPipe: pip install mediapipe")
    
    if not ADVANCED_FEATURES:
        print("⚠️  Para características avanzadas: pip install scipy scikit-learn")
    
    # Inicializar y ejecutar
    recognizer = ImprovedMediaPipeEmotionRecognition(save_data=True)
    recognizer.run_improved()