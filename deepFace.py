import cv2
import numpy as np
import os
import json
import time
from threading import Thread, Lock
import queue
from collections import deque

# Intentar importar DeepFace con manejo de errores
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace cargado correctamente")
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    print(f"❌ Error importando DeepFace: {e}")
    print("🔄 Ejecutando en modo de respaldo sin DeepFace...")
    
    class MockDeepFace:
        @staticmethod
        def verify(*args, **kwargs):
            return {"verified": False, "distance": 1.0}
        
        @staticmethod
        def analyze(*args, **kwargs):
            return [{"age": 25, "dominant_gender": "unknown", "dominant_emotion": "neutral", 
                    "emotion": {"happy": 0.1, "sad": 0.1, "angry": 0.1, "surprise": 0.1, 
                              "fear": 0.1, "disgust": 0.1, "neutral": 0.4},
                    "region": {"x": 0, "y": 0, "w": 100, "h": 100}}]
    
    DeepFace = MockDeepFace()

class EnhancedFaceEmotionRecognition:
    def __init__(self, database_path="known_faces"):
        """
        Sistema integrado de reconocimiento facial y detección de emociones
        
        Args:
            database_path: Carpeta donde se guardan las caras conocidas
        """
        self.database_path = database_path
        self.known_faces = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognition_threshold = 0.6
        self.emotion_threshold = 0.3
        self.frame_skip = 3
        self.frame_count = 0
        
        # Configuración de emociones
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
        
        # Sistema de tracking
        self.face_trackers = {}
        self.next_face_id = 0
        self.emotion_buffers = {}
        self.buffer_size = 8
        
        # Para procesamiento en hilo separado
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_lock = Lock()
        self.is_processing = False
        
        # Estadísticas
        self.emotion_history = {}  # Por persona
        self.session_start = time.time()
        
        # Crear carpeta de base de datos si no existe
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            
        self.load_known_faces()
    
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
            timestamp = int(time.time())
            image_path = os.path.join(self.database_path, f"{name}_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            
            self.known_faces[name] = {
                "image_path": image_path,
                "registered_at": timestamp
            }
            
            # Inicializar historial de emociones para esta persona
            self.emotion_history[name] = []
            
            self.save_known_faces()
            print(f"✅ Cara registrada: {name}")
            return True
            
        except Exception as e:
            print(f"❌ Error registrando cara: {e}")
            return False
    
    def process_frame_async(self, frame):
        """Procesamiento asíncrono del frame"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
    
    def process_worker(self):
        """Worker thread para procesamiento integral"""
        while True:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    with self.processing_lock:
                        if self.is_processing:
                            continue
                        self.is_processing = True
                    
                    try:
                        results = self.analyze_frame(frame)
                        if not self.result_queue.full():
                            self.result_queue.put(results)
                    except Exception as e:
                        print(f"Error en procesamiento: {e}")
                    finally:
                        with self.processing_lock:
                            self.is_processing = False
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error en worker thread: {e}")
                time.sleep(1)
    
    def analyze_frame(self, frame):
        """
        Análisis integral: reconocimiento facial + emociones
        
        Args:
            frame: Frame de la cámara
            
        Returns:
            Lista de resultados con identidad y emociones
        """
        if not DEEPFACE_AVAILABLE:
            return self.analyze_frame_fallback(frame)
        
        results = []
        
        try:
            # Usar DeepFace.analyze que incluye tanto reconocimiento como emociones
            analysis_results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace.analyze puede devolver una lista o un solo resultado
            if not isinstance(analysis_results, list):
                analysis_results = [analysis_results]
            
            for analysis in analysis_results:
                # Extraer información de la región facial
                region = analysis.get('region', {})
                x = region.get('x', 0)
                y = region.get('y', 0) 
                w = region.get('w', 100)
                h = region.get('h', 100)
                
                # Extraer región de la cara para reconocimiento
                face_region = frame[y:y+h, x:x+w] if y+h <= frame.shape[0] and x+w <= frame.shape[1] else frame
                
                # Reconocer identidad
                identity = self.recognize_identity(face_region)
                
                # Extraer emociones
                emotions = analysis.get('emotion', {})
                dominant_emotion = analysis.get('dominant_emotion', 'neutral')
                
                # Información adicional
                age = analysis.get('age', 'unknown')
                gender = analysis.get('dominant_gender', 'unknown')
                
                # Crear resultado completo
                result = {
                    "bbox": (x, y, w, h),
                    "identity": identity,
                    "emotions": emotions,
                    "dominant_emotion": dominant_emotion,
                    "emotion_confidence": emotions.get(dominant_emotion, 0),
                    "age": age,
                    "gender": gender,
                    "timestamp": time.time()
                }
                
                results.append(result)
                
                # Actualizar historial de emociones si hay identidad
                if identity["name"] != "Desconocido":
                    self.update_emotion_history(identity["name"], dominant_emotion, emotions)
        
        except Exception as e:
            print(f"Error en análisis: {e}")
            # Fallback a detección básica
            return self.analyze_frame_fallback(frame)
        
        return results
    
    def recognize_identity(self, face_region):
        """
        Reconoce la identidad de una región facial
        
        Args:
            face_region: Región de la cara extraída
            
        Returns:
            Diccionario con información de identidad
        """
        if len(self.known_faces) == 0 or face_region.size == 0:
            return {"name": "Desconocido", "confidence": 0}
        
        best_match = None
        min_distance = float('inf')
        
        for name, face_info in self.known_faces.items():
            try:
                result = DeepFace.verify(
                    img1_path=face_region,
                    img2_path=face_info["image_path"],
                    model_name="VGG-Face",
                    enforce_detection=False,
                    distance_metric="cosine",
                    silent=True
                )
                
                if result["verified"] and result["distance"] < min_distance:
                    min_distance = result["distance"]
                    best_match = name
                    
            except Exception:
                continue
        
        if best_match:
            confidence = max(0, (1 - min_distance) * 100)
            return {"name": best_match, "confidence": confidence}
        else:
            return {"name": "Desconocido", "confidence": 0}
    
    def update_emotion_history(self, name, dominant_emotion, emotions):
        """Actualiza el historial de emociones de una persona"""
        if name not in self.emotion_history:
            self.emotion_history[name] = []
        
        self.emotion_history[name].append({
            "emotion": dominant_emotion,
            "emotions": emotions,
            "timestamp": time.time()
        })
        
        # Mantener solo los últimos 100 registros por persona
        if len(self.emotion_history[name]) > 100:
            self.emotion_history[name] = self.emotion_history[name][-100:]
    
    def analyze_frame_fallback(self, frame):
        """Análisis de respaldo cuando DeepFace no está disponible"""
        results = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Simular emociones básicas
                mock_emotions = {
                    'happy': np.random.uniform(0.1, 0.3),
                    'neutral': np.random.uniform(0.4, 0.7),
                    'sad': np.random.uniform(0.05, 0.2),
                    'angry': np.random.uniform(0.05, 0.15),
                    'surprise': np.random.uniform(0.05, 0.15),
                    'fear': np.random.uniform(0.05, 0.1),
                    'disgust': np.random.uniform(0.05, 0.1)
                }
                
                # Normalizar
                total = sum(mock_emotions.values())
                mock_emotions = {k: v/total for k, v in mock_emotions.items()}
                dominant = max(mock_emotions, key=mock_emotions.get)
                
                results.append({
                    "bbox": (x, y, w, h),
                    "identity": {"name": "DeepFace no disponible", "confidence": 0},
                    "emotions": mock_emotions,
                    "dominant_emotion": dominant,
                    "emotion_confidence": mock_emotions[dominant],
                    "age": "unknown",
                    "gender": "unknown",
                    "timestamp": time.time()
                })
        
        except Exception as e:
            print(f"Error en análisis de respaldo: {e}")
        
        return results
    
    def draw_enhanced_results(self, frame, results):
        """
        Dibuja resultados mejorados con identidad y emociones
        
        Args:
            frame: Frame de la cámara
            results: Resultados del análisis
        """
        for result in results:
            x, y, w, h = result["bbox"]
            identity = result["identity"]
            name = identity["name"]
            id_confidence = identity["confidence"]
            
            dominant_emotion = result["dominant_emotion"]
            emotion_confidence = result["emotion_confidence"]
            emotions = result["emotions"]
            age = result["age"]
            gender = result["gender"]
            
            # Color basado en emoción dominante
            emotion_color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
            
            # Color del rectángulo basado en identidad
            if name != "Desconocido" and name != "DeepFace no disponible":
                border_color = (0, 255, 0)  # Verde para conocidos
                thickness = 3
            else:
                border_color = (0, 0, 255)  # Rojo para desconocidos
                thickness = 2
            
            # Dibujar rectángulo principal
            cv2.rectangle(frame, (x, y), (x+w, y+h), border_color, thickness)
            
            # Dibujar indicador de emoción (rectángulo interno)
            emotion_rect_thickness = max(1, int(emotion_confidence * 5))
            cv2.rectangle(frame, (x+5, y+5), (x+w-5, y+h-5), emotion_color, emotion_rect_thickness)
            
            # Preparar etiquetas
            if name != "Desconocido" and name != "DeepFace no disponible":
                identity_label = f"{name} ({id_confidence:.0f}%)"
            else:
                identity_label = name
            
            emotion_label = f"{dominant_emotion.title()} ({emotion_confidence:.2f})"
            info_label = f"Age: {age}, Gender: {gender}"
            
            # Calcular espacio necesario
            label_height = 90
            max_width = max(
                cv2.getTextSize(identity_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0],
                cv2.getTextSize(emotion_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0],
                cv2.getTextSize(info_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            )
            
            # Fondo para etiquetas
            cv2.rectangle(frame, (x, y-label_height), (x + max_width + 10, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y-label_height), (x + max_width + 10, y), emotion_color, 2)
            
            # Dibujar textos
            cv2.putText(frame, identity_label, (x + 5, y - 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, emotion_label, (x + 5, y - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color, 1)
            cv2.putText(frame, info_label, (x + 5, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Barra de emociones (opcional, top 3)
            if len(emotions) > 0:
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                
                bar_y_start = y + h + 10
                bar_width = w
                bar_height = 6
                
                for i, (emotion, score) in enumerate(sorted_emotions):
                    bar_y = bar_y_start + (i * 10)
                    
                    # Barra de fondo
                    cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), 
                                 (50, 50, 50), -1)
                    
                    # Barra de progreso
                    progress_width = int(bar_width * score)
                    bar_color = self.emotion_colors.get(emotion, (255, 255, 255))
                    cv2.rectangle(frame, (x, bar_y), (x + progress_width, bar_y + bar_height), 
                                 bar_color, -1)
                    
                    # Etiqueta
                    cv2.putText(frame, f"{emotion}: {score:.2f}", (x + bar_width + 5, bar_y + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, bar_color, 1)
    
    def draw_statistics(self, frame):
        """Dibuja estadísticas del sistema"""
        # Información general
        mode_text = "DeepFace Full" if DEEPFACE_AVAILABLE else "Básico"
        info_lines = [
            f"Modo: {mode_text}",
            f"Caras conocidas: {len(self.known_faces)}",
            f"Tiempo sesión: {time.time() - self.session_start:.0f}s"
        ]
        
        # Estadísticas de emociones por persona
        if self.emotion_history:
            info_lines.append("--- Emociones por persona ---")
            for name, history in self.emotion_history.items():
                if history:
                    recent_emotions = [entry["emotion"] for entry in history[-10:]]
                    most_common = max(set(recent_emotions), key=recent_emotions.count) if recent_emotions else "N/A"
                    info_lines.append(f"{name}: {most_common} (últimas 10)")
        
        # Dibujar panel
        panel_height = len(info_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (400, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 10 + panel_height), (100, 100, 100), 2)
        
        for i, line in enumerate(info_lines):
            color = (255, 255, 255) if not line.startswith("---") else (0, 255, 255)
            cv2.putText(frame, line, (15, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def save_session_data(self):
        """Guarda datos de la sesión"""
        if not self.emotion_history:
            return
        
        session_data = {
            "session_start": self.session_start,
            "session_end": time.time(),
            "known_faces": list(self.known_faces.keys()),
            "emotion_history": self.emotion_history
        }
        
        timestamp = int(time.time())
        filename = os.path.join(self.database_path, f"session_{timestamp}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            print(f"📊 Datos de sesión guardados: {filename}")
        except Exception as e:
            print(f"❌ Error guardando sesión: {e}")
    
    def run_enhanced_recognition(self):
        """Ejecuta el sistema integrado"""
        print("🚀 Sistema Integrado: Reconocimiento Facial + Detección de Emociones")
        print("="*70)
        
        if not DEEPFACE_AVAILABLE:
            print("⚠️  DeepFace no disponible - Modo simulación")
        else:
            print("✅ DeepFace activo - Análisis completo disponible")
        
        print("\n🎮 Controles:")
        if DEEPFACE_AVAILABLE:
            print("  'r': Registrar nueva cara")
        print("  'q': Salir")
        print("  'i': Información detallada")
        print("  's': Guardar datos de sesión")
        print("  'h': Mostrar/ocultar historial de emociones")
        
        # Iniciar worker thread
        worker_thread = Thread(target=self.process_worker, daemon=True)
        worker_thread.start()
        
        # Configurar cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_results = []
        registration_mode = False
        show_stats = True
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error accediendo a la cámara")
                    break
                
                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                
                # Procesar frame
                if self.frame_count % self.frame_skip == 0:
                    self.process_frame_async(frame)
                
                # Obtener resultados
                if not self.result_queue.empty():
                    last_results = self.result_queue.get()
                
                # Dibujar resultados
                self.draw_enhanced_results(frame, last_results)
                
                if show_stats:
                    self.draw_statistics(frame)
                
                # Modo registro
                if registration_mode and DEEPFACE_AVAILABLE:
                    cv2.putText(frame, "MODO REGISTRO - Presiona ESPACIO", (10, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Mostrar frame
                cv2.imshow('Sistema Integrado: Identidad + Emociones', frame)
                
                # Manejar controles
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r') and DEEPFACE_AVAILABLE:
                    registration_mode = True
                    name = input("\n👤 Nombre de la persona: ").strip()
                    if name:
                        print("📸 Posiciónate y presiona ESPACIO...")
                    else:
                        registration_mode = False
                elif key == ord(' ') and registration_mode and DEEPFACE_AVAILABLE:
                    if 'name' in locals() and name:
                        if self.register_new_face(frame, name):
                            print(f"✅ {name} registrado!")
                        registration_mode = False
                elif key == ord('i'):
                    self.print_detailed_info()
                elif key == ord('s'):
                    self.save_session_data()
                elif key == ord('h'):
                    show_stats = not show_stats
        
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_session_data()
            print("👋 Sistema finalizado")
    
    def print_detailed_info(self):
        """Imprime información detallada del sistema"""
        print("\n" + "="*50)
        print("📊 INFORMACIÓN DETALLADA DEL SISTEMA")
        print("="*50)
        print(f"🔧 DeepFace disponible: {'Sí' if DEEPFACE_AVAILABLE else 'No'}")
        print(f"👥 Personas registradas: {len(self.known_faces)}")
        
        if self.known_faces:
            print("\n👤 Lista de personas:")
            for name in self.known_faces.keys():
                print(f"   • {name}")
        
        if self.emotion_history:
            print("\n😊 Historial de emociones:")
            for name, history in self.emotion_history.items():
                if history:
                    emotions = [entry["emotion"] for entry in history]
                    emotion_counts = {}
                    for emotion in emotions:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    total = len(emotions)
                    print(f"\n   {name} ({total} detecciones):")
                    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total) * 100
                        print(f"     - {emotion}: {count} ({percentage:.1f}%)")

def main():
    """Función principal"""
    print("🚀 Iniciando Sistema Integrado...")
    
    recognizer = EnhancedFaceEmotionRecognition("mi_base_de_datos")
    
    try:
        recognizer.run_enhanced_recognition()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()