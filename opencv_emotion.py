import cv2
import numpy as np
import time
import os
from collections import Counter, deque
import json
from datetime import datetime
import threading
import queue

# Intentar importar FER con manejo de errores
try:
    from fer import FER
    FER_AVAILABLE = True
    print("✅ FER (Facial Expression Recognition) cargado correctamente")
except ImportError as e:
    FER_AVAILABLE = False
    print(f"❌ Error importando FER: {e}")
    print("🔄 Instalar con: pip install fer")
    
    # Clase mock para evitar errores
    class MockFER:
        def __init__(self, mtcnn=False):
            self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.mtcnn = mtcnn
        
        def detect_emotions(self, frame):
            return []
    
    FER = MockFER

class ImprovedEmotionRecognition:
    def __init__(self, save_data=True):
        """
        Inicializa el sistema de reconocimiento de emociones mejorado
        
        Args:
            save_data: Si guardar datos de emociones detectadas
        """
        self.save_data = save_data
        self.data_path = "emotion_data"
        
        # Configuraciones de procesamiento mejoradas
        self.process_every_n_frames = 5  # Procesar cada 5 frames en lugar de 3
        self.min_face_size = (50, 50)    # Tamaño mínimo de cara para procesar
        self.confidence_threshold = 0.3   # Umbral mínimo de confianza
        self.stabilization_frames = 8     # Frames para estabilización
        
        # Inicializar detector FER con configuraciones optimizadas
        if FER_AVAILABLE:
            self.emotion_detector = FER(mtcnn=True)
            print("🎭 Detector de emociones FER inicializado con MTCNN")
        else:
            self.emotion_detector = FER()
            print("⚠️  FER no disponible, usando modo mock")
        
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
        
        # Sistema de tracking mejorado
        self.face_trackers = {}
        self.next_face_id = 0
        self.max_tracking_distance = 100
        
        # Buffers para estabilización
        self.emotion_buffers = {}  # Un buffer por cara trackeada
        self.buffer_size = 10      # Tamaño del buffer aumentado
        
        # Estadísticas
        self.emotion_history = []
        self.detection_count = 0
        self.start_time = time.time()
        self.fps_counter = deque(maxlen=30)
        
        # Control de tiempo
        self.last_process_time = 0
        self.process_interval = 0.2  # Procesar cada 200ms mínimo
        
        # Para procesamiento asíncrono mejorado
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_lock = threading.Lock()
        self.is_processing = False
        
        # Crear carpeta para datos
        if self.save_data and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
    
    def preprocess_frame(self, frame):
        """
        Preprocesa el frame para mejorar la detección
        
        Args:
            frame: Frame original
            
        Returns:
            Frame preprocesado
        """
        # Redimensionar si es muy grande
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Mejoras en calidad de imagen
        # Reducir ruido
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # Mejorar contraste
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def detect_emotions_improved(self, frame):
        """
        Detecta emociones con validaciones mejoradas
        
        Args:
            frame: Frame preprocesado
            
        Returns:
            Lista de detecciones validadas
        """
        if not FER_AVAILABLE:
            return self.mock_detection(frame)
        
        try:
            # Detectar emociones
            emotions = self.emotion_detector.detect_emotions(frame)
            
            results = []
            for emotion_data in emotions:
                box = emotion_data["box"]
                emotions_dict = emotion_data["emotions"]
                
                # Validar tamaño de cara
                if len(box) >= 4:
                    x, y, w, h = box[:4]
                    if w < self.min_face_size[0] or h < self.min_face_size[1]:
                        continue
                
                # Encontrar emoción dominante
                dominant_emotion = max(emotions_dict, key=emotions_dict.get)
                confidence = emotions_dict[dominant_emotion]
                
                # Filtrar por confianza mínima
                if confidence < self.confidence_threshold:
                    continue
                
                # Calcular métricas de calidad
                quality_score = self.calculate_detection_quality(emotions_dict)
                
                results.append({
                    "bbox": box,
                    "emotions": emotions_dict,
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence,
                    "quality_score": quality_score,
                    "timestamp": time.time()
                })
            
            return results
            
        except Exception as e:
            print(f"Error en detección de emociones: {e}")
            return []
    
    def calculate_detection_quality(self, emotions_dict):
        """
        Calcula un score de calidad de la detección
        
        Args:
            emotions_dict: Diccionario de emociones
            
        Returns:
            Score de calidad (0-1)
        """
        # Calcular entropía para medir certeza
        probs = list(emotions_dict.values())
        entropy = -sum(p * np.log2(p + 1e-8) for p in probs if p > 0)
        max_entropy = np.log2(len(probs))
        
        # Score inverso de entropía (menos entropía = más certeza)
        certainty_score = 1 - (entropy / max_entropy)
        
        # Score basado en confianza máxima
        max_confidence = max(probs)
        
        # Combinar scores
        quality_score = (certainty_score * 0.4) + (max_confidence * 0.6)
        
        return quality_score
    
    def track_faces(self, current_detections):
        """
        Sistema de tracking de caras mejorado
        
        Args:
            current_detections: Detecciones actuales
            
        Returns:
            Detecciones con IDs de tracking
        """
        tracked_detections = []
        
        # Asignar IDs a detecciones actuales
        for detection in current_detections:
            bbox = detection["bbox"]
            if len(bbox) >= 4:
                center_x = bbox[0] + bbox[2] // 2
                center_y = bbox[1] + bbox[3] // 2
                
                # Buscar cara más cercana en trackers existentes
                best_match_id = None
                min_distance = float('inf')
                
                for face_id, tracker_data in self.face_trackers.items():
                    if time.time() - tracker_data["last_seen"] > 2.0:  # Remover trackers viejos
                        continue
                    
                    tracker_center = tracker_data["center"]
                    distance = np.sqrt(
                        (center_x - tracker_center[0])**2 + 
                        (center_y - tracker_center[1])**2
                    )
                    
                    if distance < min_distance and distance < self.max_tracking_distance:
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
                
                # Agregar ID a la detección
                detection["face_id"] = face_id
                tracked_detections.append(detection)
        
        return tracked_detections
    
    def stabilize_emotions(self, detections):
        """
        Estabiliza emociones usando buffers por cara
        
        Args:
            detections: Detecciones con tracking
            
        Returns:
            Detecciones estabilizadas
        """
        stabilized_detections = []
        
        for detection in detections:
            face_id = detection.get("face_id")
            
            if face_id is None:
                stabilized_detections.append(detection)
                continue
            
            # Inicializar buffer si no existe
            if face_id not in self.emotion_buffers:
                self.emotion_buffers[face_id] = deque(maxlen=self.buffer_size)
            
            # Agregar detección actual al buffer
            self.emotion_buffers[face_id].append(detection)
            
            # Necesitamos suficientes muestras para estabilizar
            if len(self.emotion_buffers[face_id]) < self.stabilization_frames:
                stabilized_detections.append(detection)
                continue
            
            # Calcular emociones estabilizadas
            buffer_data = list(self.emotion_buffers[face_id])
            
            # Filtrar por calidad
            quality_filtered = [d for d in buffer_data if d.get("quality_score", 0) > 0.4]
            
            if len(quality_filtered) < 3:
                quality_filtered = buffer_data  # Usar todas si no hay suficientes de calidad
            
            # Promediar emociones con pesos por calidad
            weighted_emotions = {}
            total_weight = 0
            
            for emotion in self.emotions:
                weighted_sum = 0
                for data in quality_filtered:
                    weight = data.get("quality_score", 0.5)
                    emotion_value = data["emotions"].get(emotion, 0)
                    weighted_sum += emotion_value * weight
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_emotions[emotion] = weighted_sum / total_weight
                else:
                    weighted_emotions[emotion] = 0
            
            # Crear detección estabilizada
            stabilized_detection = detection.copy()
            stabilized_detection["emotions"] = weighted_emotions
            stabilized_detection["dominant_emotion"] = max(weighted_emotions, key=weighted_emotions.get)
            stabilized_detection["confidence"] = weighted_emotions[stabilized_detection["dominant_emotion"]]
            stabilized_detection["is_stabilized"] = True
            
            stabilized_detections.append(stabilized_detection)
        
        return stabilized_detections
    
    def emotion_worker(self):
        """Worker thread optimizado para procesamiento"""
        while True:
            try:
                if not self.frame_queue.empty():
                    with self.processing_lock:
                        if self.is_processing:
                            time.sleep(0.05)
                            continue
                        self.is_processing = True
                    
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                        
                        # Preprocesar frame
                        processed_frame = self.preprocess_frame(frame)
                        
                        # Detectar emociones
                        detections = self.detect_emotions_improved(processed_frame)
                        
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
                
                time.sleep(0.05)  # Pausa más corta
                
            except Exception as e:
                print(f"Error crítico en worker: {e}")
                time.sleep(0.5)
    
    def should_process_frame(self, frame_count):
        """
        Determina si debe procesar el frame actual
        
        Args:
            frame_count: Contador de frames
            
        Returns:
            True si debe procesar
        """
        current_time = time.time()
        
        # Control por frames
        if frame_count % self.process_every_n_frames != 0:
            return False
        
        # Control por tiempo
        if current_time - self.last_process_time < self.process_interval:
            return False
        
        self.last_process_time = current_time
        return True
    
    def draw_emotions_improved(self, frame, emotion_results):
        """
        Dibuja emociones con visualización mejorada
        
        Args:
            frame: Frame de video
            emotion_results: Resultados de detección
        """
        for result in emotion_results:
            if "bbox" not in result:
                continue
            
            box = result["bbox"]
            if len(box) < 4:
                continue
            
            x, y, w, h = box[:4]
            dominant_emotion = result["dominant_emotion"]
            confidence = result["confidence"]
            emotions = result["emotions"]
            face_id = result.get("face_id", "?")
            quality_score = result.get("quality_score", 0)
            is_stabilized = result.get("is_stabilized", False)
            
            # Color basado en emoción y confianza
            base_color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            
            # Ajustar intensidad del color basado en confianza
            intensity = min(1.0, confidence * 1.5)
            color = tuple(int(c * intensity) for c in base_color)
            
            # Grosor del rectángulo basado en calidad
            thickness = 2 if quality_score > 0.6 else 1
            
            # Dibujar rectángulo principal
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Indicador de estabilización
            if is_stabilized:
                cv2.circle(frame, (x + w - 10, y + 10), 5, (0, 255, 0), -1)
            
            # Etiqueta principal mejorada
            label = f"ID{face_id}: {dominant_emotion.capitalize()}"
            confidence_text = f"Conf: {confidence:.2f}"
            quality_text = f"Qual: {quality_score:.2f}"
            
            # Fondo adaptativo para etiquetas
            label_bg_height = 75
            cv2.rectangle(frame, (x, y - label_bg_height), (x + w, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y - label_bg_height), (x + w, y), color, 2)
            
            # Textos con mejor espaciado
            cv2.putText(frame, label, (x + 5, y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, confidence_text, (x + 5, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, quality_text, (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Barra de emociones (top 3)
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            bar_width = w
            bar_height = 8
            bar_y = y + h + 10
            
            for i, (emotion, score) in enumerate(sorted_emotions):
                bar_x = x
                bar_y_current = bar_y + (i * 12)
                
                # Barra de fondo
                cv2.rectangle(frame, (bar_x, bar_y_current), (bar_x + bar_width, bar_y_current + bar_height), 
                             (50, 50, 50), -1)
                
                # Barra de progreso
                progress_width = int(bar_width * score)
                emotion_color = self.emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(frame, (bar_x, bar_y_current), (bar_x + progress_width, bar_y_current + bar_height), 
                             emotion_color, -1)
                
                # Etiqueta de emoción
                cv2.putText(frame, f"{emotion}: {score:.2f}", (bar_x + bar_width + 5, bar_y_current + 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, emotion_color, 1)
    
    def calculate_fps(self):
        """Calcula FPS en tiempo real"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                return len(self.fps_counter) / time_diff
        
        return 0
    
    def draw_statistics_improved(self, frame):
        """Dibuja estadísticas mejoradas"""
        stats = self.get_session_stats()
        fps = self.calculate_fps()
        
        # Información del sistema
        info_text = [
            f"FER: {'Active' if FER_AVAILABLE else 'Mock'} | FPS: {fps:.1f}",
            f"Tracked Faces: {len(self.face_trackers)}",
            f"Detections: {self.detection_count}",
            f"Runtime: {time.time() - self.start_time:.1f}s"
        ]
        
        if "emotion_distribution" in stats and stats["emotion_distribution"]:
            most_common = max(stats["emotion_distribution"].items(), key=lambda x: x[1])
            info_text.append(f"Most Common: {most_common[0]} ({most_common[1]})")
        
        # Panel de información
        panel_height = len(info_text) * 25 + 20
        cv2.rectangle(frame, (10, 10), (350, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 10 + panel_height), (100, 100, 100), 2)
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (15, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def mock_detection(self, frame):
        """Detección simulada mejorada"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            if w < self.min_face_size[0] or h < self.min_face_size[1]:
                continue
            
            # Simular emociones más realistas
            base_emotion = np.random.choice(self.emotions, p=[0.1, 0.05, 0.1, 0.3, 0.1, 0.05, 0.3])
            mock_emotions = {emotion: 0.1 + np.random.random() * 0.2 for emotion in self.emotions}
            mock_emotions[base_emotion] = 0.5 + np.random.random() * 0.4
            
            # Normalizar
            total = sum(mock_emotions.values())
            mock_emotions = {k: v/total for k, v in mock_emotions.items()}
            
            dominant_emotion = max(mock_emotions, key=mock_emotions.get)
            
            results.append({
                "bbox": (x, y, w, h),
                "emotions": mock_emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": mock_emotions[dominant_emotion],
                "quality_score": 0.5 + np.random.random() * 0.3,
                "timestamp": time.time()
            })
        
        return results
    
    def run_improved_recognition(self):
        """Ejecuta reconocimiento mejorado"""
        print("🎭 Iniciando reconocimiento de emociones MEJORADO...")
        
        if not FER_AVAILABLE:
            print("⚠️  FER no disponible. Modo simulación mejorado.")
        
        print("\n📋 Configuraciones:")
        print(f"   • Procesar cada {self.process_every_n_frames} frames")
        print(f"   • Intervalo mínimo: {self.process_interval}s")
        print(f"   • Umbral confianza: {self.confidence_threshold}")
        print(f"   • Frames estabilización: {self.stabilization_frames}")
        print(f"   • Tamaño mínimo cara: {self.min_face_size}")
        
        print("\n🎮 Controles:")
        print("   'q': Salir")
        print("   's': Guardar datos")
        print("   'i': Información detallada")
        print("   'r': Resetear estadísticas")
        print("   '+': Aumentar velocidad")
        print("   '-': Disminuir velocidad")
        
        # Iniciar worker
        worker_thread = threading.Thread(target=self.emotion_worker, daemon=True)
        worker_thread.start()
        
        # Configurar cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        last_results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error accediendo a la cámara")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Procesar frame si corresponde
                if self.should_process_frame(frame_count):
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                
                # Obtener resultados
                if not self.result_queue.empty():
                    last_results = self.result_queue.get()
                    self.update_statistics(last_results)
                
                # Dibujar
                self.draw_emotions_improved(frame, last_results)
                self.draw_statistics_improved(frame)
                
                # Mostrar
                cv2.imshow('Reconocimiento de Emociones MEJORADO', frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_session_data()
                elif key == ord('i'):
                    self.print_detailed_stats()
                elif key == ord('r'):
                    self.reset_statistics()
                elif key == ord('+'):
                    self.process_every_n_frames = max(1, self.process_every_n_frames - 1)
                    print(f"🔄 Procesando cada {self.process_every_n_frames} frames")
                elif key == ord('-'):
                    self.process_every_n_frames = min(10, self.process_every_n_frames + 1)
                    print(f"🔄 Procesando cada {self.process_every_n_frames} frames")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por el usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.save_data:
                self.save_session_data()
            print("👋 Reconocimiento finalizado")
    
    def update_statistics(self, emotion_results):
        """Actualiza estadísticas"""
        for result in emotion_results:
            if result.get("is_stabilized", False):  # Solo contar detecciones estabilizadas
                self.emotion_history.append({
                    "emotion": result["dominant_emotion"],
                    "confidence": result["confidence"],
                    "quality_score": result.get("quality_score", 0),
                    "timestamp": time.time()
                })
                self.detection_count += 1
    
    def get_session_stats(self):
        """Obtiene estadísticas de la sesión"""
        if not self.emotion_history:
            return {}
        
        emotions_only = [item["emotion"] for item in self.emotion_history]
        emotion_counts = Counter(emotions_only)
        
        total_time = time.time() - self.start_time
        avg_confidence = sum(item["confidence"] for item in self.emotion_history) / len(self.emotion_history)
        avg_quality = sum(item.get("quality_score", 0) for item in self.emotion_history) / len(self.emotion_history)
        
        return {
            "duration_seconds": total_time,
            "total_detections": self.detection_count,
            "emotion_distribution": dict(emotion_counts),
            "average_confidence": avg_confidence,
            "average_quality": avg_quality,
            "active_trackers": len(self.face_trackers)
        }
    
    def print_detailed_stats(self):
        """Imprime estadísticas detalladas"""
        stats = self.get_session_stats()
        
        print("\n📊 ESTADÍSTICAS DETALLADAS:")
        print(f"   ⏱️  Duración: {stats.get('duration_seconds', 0):.1f}s")
        print(f"   🎯 Detecciones: {stats.get('total_detections', 0)}")
        print(f"   📈 Confianza promedio: {stats.get('average_confidence', 0):.3f}")
        print(f"   🎨 Calidad promedio: {stats.get('average_quality', 0):.3f}")
        print(f"   👥 Caras activas: {stats.get('active_trackers', 0)}")
        
        if "emotion_distribution" in stats:
            print("\n🎭 Distribución de Emociones:")
            total = sum(stats["emotion_distribution"].values())
            for emotion, count in sorted(stats["emotion_distribution"].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"   {emotion.capitalize()}: {count} ({percentage:.1f}%)")
    
    def save_session_data(self):
        """Guarda datos de sesión"""
        if not self.save_data or not self.emotion_history:
            return
        
        session_data = {
            "start_time": self.start_time,
            "end_time": time.time(),
            "configuration": {
                "process_every_n_frames": self.process_every_n_frames,
                "confidence_threshold": self.confidence_threshold,
                "stabilization_frames": self.stabilization_frames,
                "buffer_size": self.buffer_size
            },
            "emotion_history": self.emotion_history,
            "session_stats": self.get_session_stats()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_path, f"improved_session_{timestamp}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"📊 Datos guardados en: {filename}")
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
    
    def reset_statistics(self):
        """Resetea estadísticas"""
        self.emotion_history = []
        self.detection_count = 0
        self.start_time = time.time()
        self.emotion_buffers = {}
        self.face_trackers = {}
        self.next_face_id = 0
        self.fps_counter.clear()
        print("🔄 Estadísticas y tracking reseteados")

def main():
    """Función principal mejorada"""
    print("🚀 Sistema de Reconocimiento de Emociones MEJORADO")
    print("   OpenCV + FER con optimizaciones de precisión y velocidad")
    print("="*60)
    
    # Verificar FER
    if not FER_AVAILABLE:
        print("\n⚠️  NOTA: FER no está instalado.")
        print("   Para máxima precisión, instala:")
        print("   pip install fer tensorflow")
        print("   Continuando con simulación mejorada...\n")
    
    recognizer = ImprovedEmotionRecognition(save_data=True)
    
    try:
        recognizer.run_improved_recognition()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()