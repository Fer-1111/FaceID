import cv2
import numpy as np
import os
import json
import time
from threading import Thread, Lock
import queue

# Intentar importar DeepFace con manejo de errores
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace cargado correctamente")
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    print(f"❌ Error importando DeepFace: {e}")
    print("🔄 Ejecutando en modo de respaldo sin DeepFace...")
    
    # Crear una clase mock para evitar errores
    class MockDeepFace:
        @staticmethod
        def verify(*args, **kwargs):
            return {"verified": False, "distance": 1.0}
        
        @staticmethod
        def analyze(*args, **kwargs):
            return [{"age": 25, "dominant_gender": "unknown", "dominant_emotion": "neutral", "region": {"x": 0, "y": 0, "w": 100, "h": 100}}]
    
    DeepFace = MockDeepFace()

class RealTimeFaceRecognition:
    def __init__(self, database_path="known_faces"):
        """
        Inicializa el sistema de reconocimiento facial
        
        Args:
            database_path: Carpeta donde se guardan las caras conocidas
        """
        self.database_path = database_path
        self.known_faces = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognition_threshold = 0.6
        self.frame_skip = 3  # Procesar cada 3 frames para mejor rendimiento
        self.emotion_frame_skip = 15  # Procesar emociones cada 15 frames (menos frecuente)
        self.frame_count = 0
        
        # Para procesamiento en hilo separado
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.emotion_queue = queue.Queue(maxsize=2)
        self.emotion_result_queue = queue.Queue(maxsize=2)
        self.processing_lock = Lock()
        self.is_processing = False
        
        # Crear carpeta de base de datos si no existe
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            
        self.load_known_faces()
        
        # Diccionario para mapear emociones a emojis
        self.emotion_emojis = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😮',
            'neutral': '😐'
        }
        
        # Colores para cada emoción
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Rojo
            'disgust': (0, 128, 0),    # Verde oscuro
            'fear': (128, 0, 128),     # Púrpura
            'happy': (0, 255, 255),    # Amarillo
            'sad': (255, 0, 0),        # Azul
            'surprise': (255, 165, 0), # Naranja
            'neutral': (192, 192, 192) # Gris
        }
    
    def load_known_faces(self):
        """Carga la información de caras conocidas desde archivo JSON"""
        json_path = os.path.join(self.database_path, "known_faces.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.known_faces = json.load(f)
            print(f"Cargadas {len(self.known_faces)} caras conocidas")
        else:
            self.known_faces = {}
    
    def save_known_faces(self):
        """Guarda la información de caras conocidas en archivo JSON"""
        json_path = os.path.join(self.database_path, "known_faces.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.known_faces, f, ensure_ascii=False, indent=2)
    
    def register_new_face(self, frame, name):
        """
        Registra una nueva cara en la base de datos
        
        Args:
            frame: Frame de la cámara con la cara
            name: Nombre de la persona
        """
        try:
            # Guardar imagen
            timestamp = int(time.time())
            image_path = os.path.join(self.database_path, f"{name}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            
            # Guardar información en el diccionario
            self.known_faces[name] = {
                "image_path": image_path,
                "registered_at": timestamp
            }
            
            self.save_known_faces()
            print(f"✅ Cara registrada: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Error registrando cara: {e}")
            return False
    
    def recognize_face_async(self, frame):
        """Procesamiento asíncrono de reconocimiento facial"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
    
    def analyze_emotions_async(self, frame):
        """Procesamiento asíncrono de análisis de emociones"""
        if not self.emotion_queue.full():
            self.emotion_queue.put(frame.copy())
    
    def process_recognition_worker(self):
        """Worker thread para procesamiento de reconocimiento"""
        while True:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    with self.processing_lock:
                        if self.is_processing:
                            continue
                        self.is_processing = True
                    
                    try:
                        results = self.recognize_faces(frame)
                        if not self.result_queue.full():
                            self.result_queue.put(results)
                    except Exception as e:
                        print(f"Error en reconocimiento: {e}")
                    finally:
                        with self.processing_lock:
                            self.is_processing = False
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error en worker thread: {e}")
                time.sleep(1)
    
    def process_emotion_worker(self):
        """Worker thread para procesamiento de emociones"""
        while True:
            try:
                if not self.emotion_queue.empty():
                    frame = self.emotion_queue.get()
                    
                    try:
                        emotions = self.analyze_emotions(frame)
                        if not self.emotion_result_queue.full():
                            self.emotion_result_queue.put(emotions)
                    except Exception as e:
                        print(f"Error en análisis de emociones: {e}")
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error en emotion worker thread: {e}")
                time.sleep(1)
    
    def analyze_emotions(self, frame):
        """
        Analiza las emociones en el frame usando DeepFace
        
        Args:
            frame: Frame de la cámara
            
        Returns:
            Lista con información de emociones detectadas
        """
        if not DEEPFACE_AVAILABLE:
            return []
        
        try:
            # Usar DeepFace para análisis de emociones
            results = DeepFace.analyze(
                frame, 
                actions=['emotion', 'age', 'gender'], 
                enforce_detection=False, 
                silent=True
            )
            
            # DeepFace puede retornar una lista o un dict
            if not isinstance(results, list):
                results = [results]
            
            emotions_data = []
            for result in results:
                if 'region' in result:
                    x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                    
                    emotions_data.append({
                        'bbox': (x, y, w, h),
                        'emotion': result['dominant_emotion'],
                        'age': result.get('age', 'N/A'),
                        'gender': result.get('dominant_gender', 'N/A'),
                        'emotion_scores': result.get('emotion', {})
                    })
            
            return emotions_data
            
        except Exception as e:
            print(f"Error en análisis de emociones: {e}")
            return []
    
    def recognize_faces(self, frame):
        """
        Reconoce caras en un frame
        
        Args:
            frame: Frame de la cámara
            
        Returns:
            Lista de diccionarios con información de caras detectadas
        """
        if len(self.known_faces) == 0:
            return []
        
        if not DEEPFACE_AVAILABLE:
            return self.recognize_faces_fallback(frame)
        
        results = []
        
        try:
            # Detectar caras usando Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                
                if face_region.size == 0:
                    continue
                
                best_match = None
                min_distance = float('inf')
                
                # Comparar con cada cara conocida
                for name, face_info in self.known_faces.items():
                    try:
                        result = DeepFace.verify(
                            img1_path=face_region,
                            img2_path=face_info["image_path"],
                            model_name="VGG-Face",
                            enforce_detection=False,
                            distance_metric="cosine"
                        )
                        
                        if result["verified"] and result["distance"] < min_distance:
                            min_distance = result["distance"]
                            best_match = name
                            
                    except Exception as e:
                        continue
                
                confidence = max(0, (1 - min_distance) * 100) if best_match else 0
                results.append({
                    "bbox": (x, y, w, h),
                    "name": best_match if best_match else "Desconocido",
                    "confidence": confidence
                })
        
        except Exception as e:
            print(f"Error en reconocimiento: {e}")
        
        return results
    
    def recognize_faces_fallback(self, frame):
        """Modo de respaldo para reconocimiento cuando DeepFace no está disponible"""
        results = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                results.append({
                    "bbox": (x, y, w, h),
                    "name": "DeepFace no disponible",
                    "confidence": 0
                })
        
        except Exception as e:
            print(f"Error en detección básica: {e}")
        
        return results
    
    def draw_results(self, frame, face_results, emotion_results):
        """
        Dibuja los resultados del reconocimiento y emociones en el frame
        
        Args:
            frame: Frame de la cámara
            face_results: Resultados del reconocimiento facial
            emotion_results: Resultados del análisis de emociones
        """
        # Dibujar reconocimiento facial
        for result in face_results:
            x, y, w, h = result["bbox"]
            name = result["name"]
            confidence = result["confidence"]
            
            # Color del rectángulo basado en si es conocido o no
            if name != "Desconocido":
                color = (0, 255, 0)  # Verde para conocidos
                label = f"{name} ({confidence:.1f}%)"
            else:
                color = (0, 0, 255)  # Rojo para desconocidos
                label = "Desconocido"
            
            # Dibujar rectángulo principal
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Dibujar etiqueta con fondo
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-30), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dibujar análisis de emociones
        for emotion_data in emotion_results:
            x, y, w, h = emotion_data['bbox']
            emotion = emotion_data['emotion']
            age = emotion_data['age']
            gender = emotion_data['gender']
            
            # Obtener emoji y color para la emoción
            emoji = self.emotion_emojis.get(emotion, '😐')
            emotion_color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Dibujar rectángulo para emociones (más delgado)
            cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_color, 1)
            
            # Preparar texto de emoción
            emotion_text = f"{emoji} {emotion.capitalize()}"
            info_text = f"{age} años, {gender}"
            
            # Dibujar información de emoción
            emotion_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            info_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Fondo para texto de emoción
            cv2.rectangle(frame, (x, y+h), (x + max(emotion_size[0], info_size[0]), y+h+35), emotion_color, -1)
            
            # Texto de emoción
            cv2.putText(frame, emotion_text, (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, info_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def run_real_time_recognition(self):
        """Ejecuta el reconocimiento facial y análisis de emociones en tiempo real"""
        if not DEEPFACE_AVAILABLE:
            print("⚠️  DeepFace no está disponible. Ejecutando en modo de detección básica.")
            print("   Solo se detectarán caras, pero no se reconocerán personas específicas ni emociones.")
            print("   Para funcionalidad completa, soluciona el problema de instalación de DeepFace.")
        
        print("🎥 Iniciando reconocimiento facial y análisis de emociones en tiempo real...")
        print("Controles:")
        if DEEPFACE_AVAILABLE:
            print("  'r' + Enter: Registrar nueva cara")
        print("  'e': Activar/desactivar análisis de emociones")
        print("  'q': Salir")
        print("  'i': Mostrar información")
        
        # Iniciar worker threads solo si DeepFace está disponible
        if DEEPFACE_AVAILABLE:
            recognition_worker = Thread(target=self.process_recognition_worker, daemon=True)
            emotion_worker = Thread(target=self.process_emotion_worker, daemon=True)
            recognition_worker.start()
            emotion_worker.start()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        last_face_results = []
        last_emotion_results = []
        registration_mode = False
        emotion_analysis_enabled = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo acceder a la cámara")
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            self.frame_count += 1
            
            if DEEPFACE_AVAILABLE:
                # Procesar reconocimiento facial
                if self.frame_count % self.frame_skip == 0:
                    self.recognize_face_async(frame)
                
                # Procesar análisis de emociones (menos frecuente)
                if emotion_analysis_enabled and self.frame_count % self.emotion_frame_skip == 0:
                    self.analyze_emotions_async(frame)
                
                # Obtener últimos resultados disponibles
                if not self.result_queue.empty():
                    last_face_results = self.result_queue.get()
                
                if not self.emotion_result_queue.empty():
                    last_emotion_results = self.emotion_result_queue.get()
            else:
                # Modo básico sin DeepFace
                if self.frame_count % self.frame_skip == 0:
                    last_face_results = self.recognize_faces_fallback(frame)
                    last_emotion_results = []
            
            # Dibujar resultados
            emotion_results_to_show = last_emotion_results if emotion_analysis_enabled else []
            self.draw_results(frame, last_face_results, emotion_results_to_show)
            
            # Mostrar información del sistema
            mode_text = "DeepFace" if DEEPFACE_AVAILABLE else "Básico"
            emotion_status = "ON" if emotion_analysis_enabled and DEEPFACE_AVAILABLE else "OFF"
            info_text = f"Modo: {mode_text} | Emociones: {emotion_status} | Caras: {len(self.known_faces)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if registration_mode and DEEPFACE_AVAILABLE:
                cv2.putText(frame, "MODO REGISTRO - Presiona ESPACIO", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif not DEEPFACE_AVAILABLE:
                cv2.putText(frame, "DETECCION BASICA - DeepFace no disponible", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Mostrar leyenda de emociones
            if emotion_analysis_enabled and DEEPFACE_AVAILABLE:
                y_offset = frame.shape[0] - 120
                cv2.putText(frame, "Emociones:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                for i, (emotion, emoji) in enumerate(self.emotion_emojis.items()):
                    y_pos = y_offset + 15 + (i * 12)
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    cv2.putText(frame, f"{emoji} {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Mostrar frame
            cv2.imshow('Reconocimiento Facial con Análisis de Emociones', frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r') and DEEPFACE_AVAILABLE:
                registration_mode = True
                name = input("\n👤 Ingresa el nombre de la persona: ").strip()
                if name:
                    print("📸 Posiciónate frente a la cámara y presiona ESPACIO para capturar...")
                else:
                    registration_mode = False
            elif key == ord(' ') and registration_mode and DEEPFACE_AVAILABLE:
                if 'name' in locals() and name:
                    if self.register_new_face(frame, name):
                        print(f"✅ {name} registrado exitosamente!")
                    registration_mode = False
                    del name
            elif key == ord('e'):
                emotion_analysis_enabled = not emotion_analysis_enabled
                status = "activado" if emotion_analysis_enabled else "desactivado"
                print(f"🎭 Análisis de emociones {status}")
            elif key == ord('i'):
                print(f"\n📊 Información del sistema:")
                print(f"   DeepFace disponible: {'Sí' if DEEPFACE_AVAILABLE else 'No'}")
                print(f"   Análisis de emociones: {'Activado' if emotion_analysis_enabled else 'Desactivado'}")
                print(f"   Caras registradas: {len(self.known_faces)}")
                for name in self.known_faces.keys():
                    print(f"   - {name}")
                print(f"   Emociones detectables: {', '.join(self.emotion_emojis.keys())}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Reconocimiento facial y análisis de emociones finalizado")

def main():
    """Función principal"""
    print("🚀 Iniciando sistema de reconocimiento facial con análisis de emociones...")
    
    recognizer = RealTimeFaceRecognition("mi_base_de_datos")
    
    try:
        recognizer.run_real_time_recognition()
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()