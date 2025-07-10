import cv2
import numpy as np
import time
import os
from collections import Counter
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
        def __init__(self, mtcnn=False):  # Aceptar parámetros para evitar errores
            self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.mtcnn = mtcnn
        
        def detect_emotions(self, frame):
            return []
    
    FER = MockFER

class OpenCVEmotionRecognition:
    def __init__(self, save_data=True):
        """
        Inicializa el sistema de reconocimiento de emociones con OpenCV + FER
        
        Args:
            save_data: Si guardar datos de emociones detectadas
        """
        self.save_data = save_data
        self.data_path = "emotion_data"
        
        # Inicializar detector FER
        if FER_AVAILABLE:
            self.emotion_detector = FER(mtcnn=True)  # Usar MTCNN para mejor detección
            print("🎭 Detector de emociones FER inicializado con MTCNN")
        else:
            self.emotion_detector = FER()
            print("⚠️  FER no disponible, usando modo mock")
        
        # Configuraciones
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
        
        # Estadísticas
        self.emotion_history = []
        self.detection_count = 0
        self.start_time = time.time()
        
        # Para procesamiento asíncrono
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing = False
        
        # Para suavizado de emociones
        self.emotion_buffer = []
        self.buffer_size = 5
        
        # Crear carpeta para datos
        if self.save_data and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
    
    def detect_emotions(self, frame):
        """
        Detecta emociones en un frame
        
        Args:
            frame: Frame de la cámara
            
        Returns:
            Lista de diccionarios con emociones detectadas
        """
        if not FER_AVAILABLE:
            return self.mock_detection(frame)
        
        try:
            # Detectar emociones usando FER
            emotions = self.emotion_detector.detect_emotions(frame)
            
            results = []
            for emotion_data in emotions:
                box = emotion_data["box"]
                emotions_dict = emotion_data["emotions"]
                
                # Encontrar emoción dominante
                dominant_emotion = max(emotions_dict, key=emotions_dict.get)
                confidence = emotions_dict[dominant_emotion]
                
                results.append({
                    "bbox": box,
                    "emotions": emotions_dict,
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence
                })
            
            return results
            
        except Exception as e:
            print(f"Error en detección de emociones: {e}")
            return []
    
    def mock_detection(self, frame):
        """Detección simulada cuando FER no está disponible"""
        # Usar OpenCV para detectar rostros básicos
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            # Simular emociones aleatorias
            mock_emotions = {emotion: np.random.random() for emotion in self.emotions}
            dominant_emotion = max(mock_emotions, key=mock_emotions.get)
            
            results.append({
                "bbox": (x, y, w, h),
                "emotions": mock_emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": mock_emotions[dominant_emotion]
            })
        
        return results
    
    def smooth_emotions(self, current_emotions):
        """
        Suaviza las emociones detectadas usando un buffer
        
        Args:
            current_emotions: Lista de emociones actuales
            
        Returns:
            Lista de emociones suavizadas
        """
        if not current_emotions:
            return current_emotions
        
        # Agregar al buffer
        self.emotion_buffer.append(current_emotions)
        
        # Mantener tamaño del buffer
        if len(self.emotion_buffer) > self.buffer_size:
            self.emotion_buffer.pop(0)
        
        # Suavizar cada detección
        smoothed_results = []
        for i, current_detection in enumerate(current_emotions):
            if len(self.emotion_buffer) < 2:
                smoothed_results.append(current_detection)
                continue
            
            # Promediar emociones del buffer
            averaged_emotions = {}
            for emotion in self.emotions:
                values = []
                for buffer_frame in self.emotion_buffer:
                    if i < len(buffer_frame):
                        values.append(buffer_frame[i]["emotions"].get(emotion, 0))
                
                if values:
                    averaged_emotions[emotion] = sum(values) / len(values)
                else:
                    averaged_emotions[emotion] = 0
            
            # Encontrar emoción dominante suavizada
            dominant_emotion = max(averaged_emotions, key=averaged_emotions.get)
            
            smoothed_result = current_detection.copy()
            smoothed_result["emotions"] = averaged_emotions
            smoothed_result["dominant_emotion"] = dominant_emotion
            smoothed_result["confidence"] = averaged_emotions[dominant_emotion]
            
            smoothed_results.append(smoothed_result)
        
        return smoothed_results
    
    def process_frame_async(self, frame):
        """Procesa frame de forma asíncrona"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
    
    def emotion_worker(self):
        """Worker thread para procesamiento de emociones"""
        while True:
            try:
                if not self.frame_queue.empty() and not self.processing:
                    frame = self.frame_queue.get()
                    self.processing = True
                    
                    try:
                        results = self.detect_emotions(frame)
                        smoothed_results = self.smooth_emotions(results)
                        
                        if not self.result_queue.full():
                            self.result_queue.put(smoothed_results)
                    
                    except Exception as e:
                        print(f"Error en worker: {e}")
                    finally:
                        self.processing = False
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error crítico en worker: {e}")
                time.sleep(1)
    
    def draw_emotions(self, frame, emotion_results):
        """
        Dibuja las emociones detectadas en el frame
        
        Args:
            frame: Frame de la cámara
            emotion_results: Resultados de detección de emociones
        """
        for result in emotion_results:
            if "bbox" in result:
                box = result["bbox"]
                if len(box) == 4:
                    x, y, w, h = box
                else:
                    continue
            else:
                continue
            
            dominant_emotion = result["dominant_emotion"]
            confidence = result["confidence"]
            emotions = result["emotions"]
            
            # Color basado en la emoción dominante
            color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            
            # Dibujar rectángulo principal
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Etiqueta principal
            label = f"{dominant_emotion.capitalize()}: {confidence:.2f}"
            
            # Fondo para la etiqueta
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostrar top 3 emociones
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for i, (emotion, score) in enumerate(sorted_emotions):
                emotion_text = f"{emotion}: {score:.2f}"
                y_offset = y + h + 20 + (i * 20)
                
                # Fondo semi-transparente
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y_offset - 15), (x + 150, y_offset + 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Texto
                emotion_color = self.emotion_colors.get(emotion, (255, 255, 255))
                cv2.putText(frame, emotion_text, (x + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, emotion_color, 1)
    
    def update_statistics(self, emotion_results):
        """Actualiza estadísticas de emociones"""
        for result in emotion_results:
            dominant_emotion = result["dominant_emotion"]
            confidence = result["confidence"]
            
            self.emotion_history.append({
                "emotion": dominant_emotion,
                "confidence": confidence,
                "timestamp": time.time()
            })
            
            self.detection_count += 1
    
    def save_session_data(self):
        """Guarda datos de la sesión"""
        if not self.save_data or not self.emotion_history:
            return
        
        session_data = {
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_detections": self.detection_count,
            "emotion_history": self.emotion_history,
            "session_stats": self.get_session_stats()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_path, f"session_{timestamp}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"📊 Datos guardados en: {filename}")
        except Exception as e:
            print(f"❌ Error guardando datos: {e}")
    
    def get_session_stats(self):
        """Obtiene estadísticas de la sesión actual"""
        if not self.emotion_history:
            return {}
        
        emotions_only = [item["emotion"] for item in self.emotion_history]
        emotion_counts = Counter(emotions_only)
        
        total_time = time.time() - self.start_time
        avg_fps = self.detection_count / total_time if total_time > 0 else 0
        
        return {
            "duration_seconds": total_time,
            "average_fps": avg_fps,
            "emotion_distribution": dict(emotion_counts),
            "most_common_emotion": emotion_counts.most_common(1)[0] if emotion_counts else None
        }
    
    def draw_statistics(self, frame):
        """Dibuja estadísticas en tiempo real"""
        stats = self.get_session_stats()
        
        # Información básica
        runtime = time.time() - self.start_time
        fps = self.detection_count / runtime if runtime > 0 else 0
        
        info_text = [
            f"FER Model: {'Active' if FER_AVAILABLE else 'Mock Mode'}",
            f"Runtime: {runtime:.1f}s",
            f"Detections: {self.detection_count}",
            f"Avg FPS: {fps:.1f}",
        ]
        
        # Mostrar distribución de emociones
        if "emotion_distribution" in stats:
            most_common = stats.get("most_common_emotion")
            if most_common:
                info_text.append(f"Most common: {most_common[0]} ({most_common[1]})")
        
        # Dibujar información
        y_offset = 30
        for text in info_text:
            # Fondo
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (10, y_offset - 20), (10 + text_size[0] + 10, y_offset + 5), (0, 0, 0), -1)
            
            # Texto
            cv2.putText(frame, text, (15, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    def run_emotion_recognition(self):
        """Ejecuta el reconocimiento de emociones en tiempo real"""
        print("🎭 Iniciando reconocimiento de emociones con OpenCV + FER...")
        
        if not FER_AVAILABLE:
            print("⚠️  FER no está disponible. Ejecutando en modo simulación.")
            print("   Para funcionalidad completa, instala: pip install fer")
        
        print("\nControles:")
        print("  'q': Salir")
        print("  's': Guardar estadísticas")
        print("  'i': Mostrar información detallada")
        print("  'r': Resetear estadísticas")
        
        # Iniciar worker thread
        if FER_AVAILABLE:
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
                    print("❌ No se pudo acceder a la cámara")
                    break
                
                # Voltear frame para efecto espejo
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Procesar cada 3 frames para mejor rendimiento
                if frame_count % 3 == 0:
                    if FER_AVAILABLE:
                        self.process_frame_async(frame)
                    else:
                        last_results = self.detect_emotions(frame)
                        self.update_statistics(last_results)
                
                # Obtener últimos resultados
                if FER_AVAILABLE and not self.result_queue.empty():
                    last_results = self.result_queue.get()
                    self.update_statistics(last_results)
                
                # Dibujar resultados
                self.draw_emotions(frame, last_results)
                self.draw_statistics(frame)
                
                # Mostrar frame
                cv2.imshow('Reconocimiento de Emociones - OpenCV + FER', frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_session_data()
                elif key == ord('i'):
                    self.print_detailed_stats()
                elif key == ord('r'):
                    self.reset_statistics()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por el usuario")
        
        finally:
            # Limpiar recursos
            cap.release()
            cv2.destroyAllWindows()
            
            # Guardar datos finales
            if self.save_data:
                self.save_session_data()
            
            print("👋 Reconocimiento de emociones finalizado")
    
    def print_detailed_stats(self):
        """Imprime estadísticas detalladas"""
        stats = self.get_session_stats()
        
        print("\n📊 Estadísticas Detalladas:")
        print(f"   Duración: {stats.get('duration_seconds', 0):.1f} segundos")
        print(f"   FPS promedio: {stats.get('average_fps', 0):.2f}")
        print(f"   Total detecciones: {self.detection_count}")
        
        if "emotion_distribution" in stats:
            print("\n🎭 Distribución de Emociones:")
            for emotion, count in stats["emotion_distribution"].items():
                percentage = (count / self.detection_count) * 100
                print(f"   {emotion.capitalize()}: {count} ({percentage:.1f}%)")
    
    def reset_statistics(self):
        """Resetea las estadísticas"""
        self.emotion_history = []
        self.detection_count = 0
        self.start_time = time.time()
        self.emotion_buffer = []
        print("🔄 Estadísticas reseteadas")

def main():
    """Función principal"""
    print("🚀 Iniciando sistema de reconocimiento de emociones...")
    print("   Usando OpenCV + FER (Facial Expression Recognition)")
    
    # Verificar instalación de FER
    if not FER_AVAILABLE:
        print("\n⚠️  ADVERTENCIA: FER no está instalado.")
        print("   Para funcionalidad completa, ejecuta:")
        print("   pip install fer")
        print("   pip install tensorflow")
        print("\n   El programa continuará en modo simulación...\n")
    
    recognizer = OpenCVEmotionRecognition(save_data=True)
    
    try:
        recognizer.run_emotion_recognition()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()