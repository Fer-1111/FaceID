# mediaPipe.py
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

# Intentar importar MediaPipe con manejo de errores
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
        """
        Inicializa el sistema de reconocimiento de emociones con MediaPipe
        
        Args:
            save_data: Si guardar datos de emociones detectadas
        """
        self.save_data = save_data
        self.data_path = "emotion_data_mediapipe"
        
        # Configuraciones de procesamiento
        self.process_every_n_frames = 2  # MediaPipe es más rápido
        self.confidence_threshold = 0.5
        self.stabilization_frames = 5
        
        # Inicializar MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # Configurar Face Mesh con parámetros optimizados
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
        
        # Configuraciones de emociones
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Rojo
            'disgust': (0, 128, 0),    # Verde oscuro
            'fear': (128, 0, 128),     # Púrpura
            'happy': (0, 255, 255),    # Amarillo
            'sad': (255, 0, 0),        # Azul
            'surprise': (0, 165, 255), # Naranja
            'neutral': (128, 128, 128) # Gris
        }
        
        # Landmark indices para diferentes características faciales
        self.landmark_indices = {
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'eyes': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'eyebrows': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107, 276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 237, 238, 239, 240, 241, 242]
        }
        
        # Sistema de tracking
        self.face_trackers = {}
        self.next_face_id = 0
        
        # Buffers para estabilización
        self.emotion_buffers = {}
        self.buffer_size = 8
        
        # Estadísticas
        self.emotion_history = []
        self.detection_count = 0
        self.start_time = time.time()
        self.fps_counter = deque(maxlen=30)
        
        # Control de tiempo
        self.last_process_time = 0
        self.process_interval = 0.1  # MediaPipe es más rápido
        
        # Para procesamiento asíncrono
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_lock = threading.Lock()
        self.is_processing = False
        
        # Variables de control
        self.show_landmarks = True
        self.running = True
        
        # Crear carpeta para datos
        if self.save_data and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
    
    def calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana entre dos puntos"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def analyze_facial_features(self, landmarks, image_width, image_height):
        """
        Analiza características faciales para determinar emociones
        
        Args:
            landmarks: Landmarks faciales de MediaPipe
            image_width: Ancho de la imagen
            image_height: Alto de la imagen
            
        Returns:
            Diccionario con características faciales
        """
        features = {}
        
        # Convertir landmarks a coordenadas de píxeles
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            landmark_points.append((x, y))
        
        # Análisis de boca
        mouth_points = [landmark_points[i] for i in self.landmark_indices['mouth']]
        mouth_width = self.calculate_distance(mouth_points[0], mouth_points[6])
        mouth_height = self.calculate_distance(mouth_points[3], mouth_points[9])
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Análisis de ojos
        left_eye_points = [landmark_points[i] for i in [33, 7, 163, 144, 145, 153]]
        right_eye_points = [landmark_points[i] for i in [362, 382, 381, 380, 374, 373]]
        
        left_eye_ratio = self.calculate_eye_ratio(left_eye_points)
        right_eye_ratio = self.calculate_eye_ratio(right_eye_points)
        eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        # Análisis de cejas
        left_eyebrow_points = [landmark_points[i] for i in [46, 53, 52, 65, 55, 70]]
        right_eyebrow_points = [landmark_points[i] for i in [276, 283, 282, 295, 285, 300]]
        
        eyebrow_height = self.calculate_eyebrow_height(left_eyebrow_points, right_eyebrow_points, landmark_points)
        
        # Análisis de mejillas (usando puntos de referencia)
        cheek_points = [landmark_points[i] for i in [116, 117, 118, 119, 120, 345, 346, 347, 348, 349]]
        cheek_curvature = self.calculate_cheek_curvature(cheek_points)
        
        features = {
            'mouth_ratio': mouth_ratio,
            'mouth_width': mouth_width,
            'eye_ratio': eye_ratio,
            'eyebrow_height': eyebrow_height,
            'cheek_curvature': cheek_curvature
        }
        
        return features
    
    def calculate_eye_ratio(self, eye_points):
        """Calcula ratio de apertura del ojo"""
        if len(eye_points) < 6:
            return 0.5
        
        # Distancia vertical del ojo
        vertical_dist = (self.calculate_distance(eye_points[1], eye_points[5]) + 
                        self.calculate_distance(eye_points[2], eye_points[4])) / 2
        
        # Distancia horizontal del ojo
        horizontal_dist = self.calculate_distance(eye_points[0], eye_points[3])
        
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.5
    
    def calculate_eyebrow_height(self, left_eyebrow, right_eyebrow, all_points):
        """Calcula altura relativa de las cejas"""
        if len(left_eyebrow) < 3 or len(right_eyebrow) < 3:
            return 0.5
        
        # Punto central de la cara (nariz)
        nose_bridge = all_points[6]  # Punto del puente nasal
        
        # Altura promedio de cejas
        left_avg_y = sum(point[1] for point in left_eyebrow) / len(left_eyebrow)
        right_avg_y = sum(point[1] for point in right_eyebrow) / len(right_eyebrow)
        eyebrow_avg_y = (left_avg_y + right_avg_y) / 2
        
        # Normalizar respecto a la posición de la nariz
        relative_height = (nose_bridge[1] - eyebrow_avg_y) / nose_bridge[1] if nose_bridge[1] > 0 else 0.5
        
        return max(0, min(1, relative_height))
    
    def calculate_cheek_curvature(self, cheek_points):
        """Calcula curvatura de mejillas (indicador de sonrisa)"""
        if len(cheek_points) < 5:
            return 0.5
        
        # Calcular curvatura promedio
        curvatures = []
        for i in range(2, len(cheek_points) - 2):
            # Usar 3 puntos para calcular curvatura
            p1, p2, p3 = cheek_points[i-1], cheek_points[i], cheek_points[i+1]
            
            # Calcular ángulo
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Producto punto normalizado
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp valores
                curvatures.append(cos_angle)
        
        return sum(curvatures) / len(curvatures) if curvatures else 0.5
    
    def features_to_emotion(self, features):
        """
        Convierte características faciales a emociones usando reglas heurísticas
        
        Args:
            features: Diccionario de características faciales
            
        Returns:
            Diccionario de probabilidades de emociones
        """
        emotions = {}
        
        # Reglas heurísticas basadas en investigación de expresiones faciales
        mouth_ratio = features['mouth_ratio']
        eye_ratio = features['eye_ratio']
        eyebrow_height = features['eyebrow_height']
        cheek_curvature = features['cheek_curvature']
        
        # Felicidad - boca amplia, mejillas elevadas, ojos ligeramente cerrados
        happiness = (
            (mouth_ratio * 0.3) +
            (max(0, cheek_curvature - 0.5) * 0.4) +
            (max(0, 0.7 - eye_ratio) * 0.3)
        )
        
        # Tristeza - boca hacia abajo, cejas bajas, ojos ligeramente cerrados
        sadness = (
            (max(0, 0.3 - mouth_ratio) * 0.4) +
            (max(0, 0.4 - eyebrow_height) * 0.3) +
            (max(0, 0.6 - eye_ratio) * 0.3)
        )
        
        # Enojo - cejas bajas, boca tensa, ojos abiertos
        anger = (
            (max(0, 0.3 - eyebrow_height) * 0.4) +
            (max(0, eye_ratio - 0.6) * 0.3) +
            (max(0, 0.5 - cheek_curvature) * 0.3)
        )
        
        # Sorpresa - cejas altas, ojos muy abiertos, boca abierta
        surprise = (
            (max(0, eyebrow_height - 0.7) * 0.4) +
            (max(0, eye_ratio - 0.7) * 0.3) +
            (max(0, mouth_ratio - 0.4) * 0.3)
        )
        
        # Miedo - cejas altas, ojos muy abiertos, boca ligeramente abierta
        fear = (
            (max(0, eyebrow_height - 0.6) * 0.4) +
            (max(0, eye_ratio - 0.6) * 0.4) +
            (max(0, mouth_ratio - 0.2) * 0.2)
        )
        
        # Disgusto - cejas bajas, boca hacia abajo, mejillas tensas
        disgust = (
            (max(0, 0.4 - eyebrow_height) * 0.3) +
            (max(0, 0.3 - mouth_ratio) * 0.4) +
            (max(0, 0.4 - cheek_curvature) * 0.3)
        )
        
        # Neutral - valores medios
        neutral = 1 - max(happiness, sadness, anger, surprise, fear, disgust)
        neutral = max(0.1, neutral)  # Valor mínimo para neutral
        
        emotions = {
            'happy': happiness,
            'sad': sadness,
            'angry': anger,
            'surprise': surprise,
            'fear': fear,
            'disgust': disgust,
            'neutral': neutral
        }
        
        # Normalizar probabilidades
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def detect_emotions_mediapipe(self, frame):
        """
        Detecta emociones usando MediaPipe
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Lista de detecciones
        """
        if not MEDIAPIPE_AVAILABLE:
            return self.mock_detection(frame)
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            detections = []
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calcular bounding box
                    h, w = frame.shape[:2]
                    x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                    y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Expandir bounding box ligeramente
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    # Analizar características faciales
                    features = self.analyze_facial_features(face_landmarks, w, h)
                    
                    # Convertir a emociones
                    emotions = self.features_to_emotion(features)
                    
                    # Encontrar emoción dominante
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    
                    # Calcular score de calidad
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
    
    def calculate_quality_score(self, features, confidence):
        """Calcula score de calidad de la detección"""
        # Considerar la coherencia de las características
        feature_values = list(features.values())
        
        # Penalizar valores extremos (pueden ser ruido)
        extreme_penalty = sum(1 for v in feature_values if v < 0.1 or v > 0.9)
        extreme_penalty = extreme_penalty / len(feature_values)
        
        # Score basado en confianza y coherencia
        quality_score = confidence * (1 - extreme_penalty * 0.3)
        
        return max(0.1, min(1.0, quality_score))
    
    def track_faces(self, detections):
        """Sistema de tracking para MediaPipe"""
        tracked_detections = []
        
        for detection in detections:
            bbox = detection["bbox"]
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            
            # Buscar tracker más cercano
            best_match_id = None
            min_distance = float('inf')
            
            for face_id, tracker_data in self.face_trackers.items():
                if time.time() - tracker_data["last_seen"] > 1.5:
                    continue
                
                tracker_center = tracker_data["center"]
                distance = np.sqrt(
                    (center_x - tracker_center[0])**2 + 
                    (center_y - tracker_center[1])**2
                )
                
                if distance < min_distance and distance < 80:
                    min_distance = distance
                    best_match_id = face_id
            
            # Asignar ID
            if best_match_id is not None:
                face_id = best_match_id
            else:
                face_id = self.next_face_id
                self.next_face_id += 1
            
            # Actualizar tracker
            self.face_trackers[face_id] = {
                "center": (center_x, center_y),
                "bbox": bbox,
                "last_seen": time.time()
            }
            
            detection["face_id"] = face_id
            tracked_detections.append(detection)
        
        return tracked_detections
    
    def stabilize_emotions(self, detections):
        """Estabiliza emociones usando buffers"""
        stabilized_detections = []
        
        for detection in detections:
            face_id = detection.get("face_id")
            
            if face_id is None:
                stabilized_detections.append(detection)
                continue
            
            # Inicializar buffer
            if face_id not in self.emotion_buffers:
                self.emotion_buffers[face_id] = deque(maxlen=self.buffer_size)
            
            # Agregar al buffer
            self.emotion_buffers[face_id].append(detection)
            
            # Estabilizar si hay suficientes muestras
            if len(self.emotion_buffers[face_id]) < self.stabilization_frames:
                stabilized_detections.append(detection)
                continue
            
            # Promediar emociones
            buffer_data = list(self.emotion_buffers[face_id])
            
            averaged_emotions = {}
            for emotion in self.emotions:
                values = [data["emotions"].get(emotion, 0) for data in buffer_data]
                averaged_emotions[emotion] = sum(values) / len(values)
            
            # Crear detección estabilizada
            stabilized_detection = detection.copy()
            stabilized_detection["emotions"] = averaged_emotions
            stabilized_detection["dominant_emotion"] = max(averaged_emotions, key=averaged_emotions.get)
            stabilized_detection["confidence"] = averaged_emotions[stabilized_detection["dominant_emotion"]]
            stabilized_detection["is_stabilized"] = True
            
            stabilized_detections.append(stabilized_detection)
        
        return stabilized_detections
    
    def emotion_worker(self):
        """Worker thread para procesamiento"""
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
                        
                        # Detectar emociones
                        detections = self.detect_emotions_mediapipe(frame)
                        
                        # Aplicar tracking
                        tracked_detections = self.track_faces(detections)
                        
                        # Estabilizar emociones
                        stabilized_detections = self.stabilize_emotions(tracked_detections)
                        
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
                
                time.sleep(0.02)  # MediaPipe puede procesar más rápido
                
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
    
    def draw_emotions_mediapipe(self, frame, results):
        """Dibuja resultados de MediaPipe"""
        for result in results:
            bbox = result["bbox"]
            x, y, w, h = bbox
            
            dominant_emotion = result["dominant_emotion"]
            confidence = result["confidence"]
            emotions = result["emotions"]
            face_id = result.get("face_id", "?")
            quality_score = result.get("quality_score", 0)
            is_stabilized = result.get("is_stabilized", False)
            
            # Color y grosor
            color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            thickness = 2 if quality_score > 0.6 else 1
            
            # Dibujar bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Indicador de estabilización
            if is_stabilized:
                cv2.circle(frame, (x + w - 10, y + 10), 5, (0, 255, 0), -1)
            
            # Dibujar landmarks si disponibles
            if self.show_landmarks and "landmarks" in result and MEDIAPIPE_AVAILABLE:
                self.mp_drawing.draw_landmarks(
                    frame, result["landmarks"], self.mp_face_mesh.FACEMESH_CONTOURS,
                    None, self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # Etiquetas
            label = f"MP-ID{face_id}: {dominant_emotion.capitalize()}"
            conf_text = f"Conf: {confidence:.2f}"
            qual_text = f"Qual: {quality_score:.2f}"
            
            # Fondo para etiquetas
            cv2.rectangle(frame, (x, y - 75), (x + w, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y - 75), (x + w, y), color, 2)
            
            # Textos
            cv2.putText(frame, label, (x + 5, y - 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, conf_text, (x + 5, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, qual_text, (x + 5, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def calculate_fps(self):
        """Calcula FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                return len(self.fps_counter) / time_diff
        return 0
    
    def draw_statistics(self, frame):
        """Dibuja estadísticas"""
        fps = self.calculate_fps()
        
        info_text = [
            f"MediaPipe: {'Active' if MEDIAPIPE_AVAILABLE else 'Mock'} | FPS: {fps:.1f}",
            f"Tracked Faces: {len(self.face_trackers)}",
            f"Detections: {self.detection_count}",
            f"Runtime: {time.time() - self.start_time:.1f}s"
        ]
        
        # Panel
        panel_height = len(info_text) * 25 + 20
        cv2.rectangle(frame, (10, 10), (400, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 10 + panel_height), (100, 100, 100), 2)
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (15, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def mock_detection(self, frame):
        """Detección mock cuando MediaPipe no está disponible"""
        # Usar detección básica de OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            # Simular emociones
            mock_emotions = {emotion: np.random.random() * 0.3 for emotion in self.emotions}
            dominant = np.random.choice(self.emotions)
            mock_emotions[dominant] = 0.4 + np.random.random() * 0.5
            
            # Normalizar
            total = sum(mock_emotions.values())
            mock_emotions = {k: v/total for k, v in mock_emotions.items()}
            
            results.append({
                "bbox": (x, y, w, h),
                "emotions": mock_emotions,
                "dominant_emotion": dominant,
                "confidence": mock_emotions[dominant],
                "quality_score": 0.5,
                "timestamp": time.time()
            })
        
        return results
    
    def save_emotion_data(self):
        """Guarda datos de emociones"""
        if not self.save_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_path}/emotions_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "total_detections": self.detection_count,
            "runtime": time.time() - self.start_time,
            "emotion_history": self.emotion_history[-100:],  # Últimas 100 detecciones
            "face_trackers": len(self.face_trackers)
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Datos guardados en {filename}")
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
    
    def reset_statistics(self):
        """Resetea estadísticas"""
        self.detection_count = 0
        self.start_time = time.time()
        self.emotion_history = []
        self.fps_counter.clear()
        self.face_trackers.clear()
        self.emotion_buffers.clear()
        self.next_face_id = 0
        print("📊 Estadísticas reseteadas")
    
    def cleanup(self):
        """Limpia recursos y guarda si corresponde"""
        self.running = False
        if MEDIAPIPE_AVAILABLE and self.face_mesh:
            self.face_mesh.close()
        self.save_emotion_data()
        print("🧹 Recursos liberados")

    def run(self):
        """Inicia la captura de video y procesamiento"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error al abrir la cámara")
            return

        # Iniciar worker thread
        threading.Thread(target=self.emotion_worker, daemon=True).start()
        print("🎥 Captura iniciada. Presiona 'q' para salir.")

        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No se pudo leer el frame")
                break

            frame_count += 1

            # Enviar frame a procesamiento si corresponde
            if self.should_process_frame(frame_count):
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

            # Mostrar resultados si hay
            if not self.result_queue.empty():
                detections = self.result_queue.get()
                self.detection_count += len(detections)
                self.emotion_history.extend([d["dominant_emotion"] for d in detections])
                self.draw_emotions_mediapipe(frame, detections)

            self.draw_statistics(frame)
            cv2.imshow("MediaPipe Emotion Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🚪 Saliendo...")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.cleanup()

# Ejecutar si es módulo principal
if __name__ == "__main__":
    recognizer = MediaPipeEmotionRecognition(save_data=True)
    recognizer.run()
