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
        self.frame_count = 0
        
        # Para procesamiento en hilo separado
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_lock = Lock()
        self.is_processing = False
        
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
                
                time.sleep(0.1)  # Pequeña pausa para evitar uso excesivo de CPU
                
            except Exception as e:
                print(f"Error en worker thread: {e}")
                time.sleep(1)
    
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
            # Modo de respaldo sin DeepFace
            return self.recognize_faces_fallback(frame)
        
        results = []
        
        try:
            # Detectar caras usando Haar Cascade (más rápido para detección inicial)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Extraer región de la cara
                face_region = frame[y:y+h, x:x+w]
                
                if face_region.size == 0:
                    continue
                
                best_match = None
                min_distance = float('inf')
                
                # Comparar con cada cara conocida
                for name, face_info in self.known_faces.items():
                    try:
                        # Usar DeepFace para verificación
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
                
                # Agregar resultado
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
        """
        Modo de respaldo para reconocimiento cuando DeepFace no está disponible
        Usa solo detección de caras con Haar Cascade
        """
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
    
    def draw_results(self, frame, results):
        """
        Dibuja los resultados del reconocimiento en el frame
        
        Args:
            frame: Frame de la cámara
            results: Resultados del reconocimiento
        """
        for result in results:
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
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Dibujar etiqueta con fondo
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-30), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run_real_time_recognition(self):
        """Ejecuta el reconocimiento facial en tiempo real"""
        if not DEEPFACE_AVAILABLE:
            print("⚠️  DeepFace no está disponible. Ejecutando en modo de detección básica.")
            print("   Solo se detectarán caras, pero no se reconocerán personas específicas.")
            print("   Para reconocimiento completo, soluciona el problema de instalación de DeepFace.")
        
        print("🎥 Iniciando reconocimiento facial en tiempo real...")
        print("Controles:")
        if DEEPFACE_AVAILABLE:
            print("  'r' + Enter: Registrar nueva cara")
        print("  'q': Salir")
        print("  'i': Mostrar información")
        
        # Iniciar worker thread solo si DeepFace está disponible
        if DEEPFACE_AVAILABLE:
            worker_thread = Thread(target=self.process_recognition_worker, daemon=True)
            worker_thread.start()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        last_results = []
        registration_mode = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo acceder a la cámara")
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            self.frame_count += 1
            
            if DEEPFACE_AVAILABLE:
                # Procesar reconocimiento cada ciertos frames
                if self.frame_count % self.frame_skip == 0:
                    self.recognize_face_async(frame)
                
                # Obtener últimos resultados disponibles
                if not self.result_queue.empty():
                    last_results = self.result_queue.get()
            else:
                # Modo básico sin DeepFace
                if self.frame_count % self.frame_skip == 0:
                    last_results = self.recognize_faces_fallback(frame)
            
            # Dibujar resultados
            self.draw_results(frame, last_results)
            
            # Mostrar información del sistema
            mode_text = "DeepFace" if DEEPFACE_AVAILABLE else "Básico"
            info_text = f"Modo: {mode_text} | Caras conocidas: {len(self.known_faces)} | FPS: ~{30//self.frame_skip}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if registration_mode and DEEPFACE_AVAILABLE:
                cv2.putText(frame, "MODO REGISTRO - Presiona ESPACIO", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif not DEEPFACE_AVAILABLE:
                cv2.putText(frame, "DETECCION BASICA - DeepFace no disponible", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Reconocimiento Facial en Tiempo Real', frame)
            
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
            elif key == ord('i'):
                print(f"\n📊 Información del sistema:")
                print(f"   DeepFace disponible: {'Sí' if DEEPFACE_AVAILABLE else 'No'}")
                print(f"   Caras registradas: {len(self.known_faces)}")
                for name in self.known_faces.keys():
                    print(f"   - {name}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Reconocimiento facial finalizado")

def main():
    """Función principal"""
    print("🚀 Iniciando sistema de reconocimiento facial...")
    
    recognizer = RealTimeFaceRecognition("mi_base_de_datos")
    
    try:
        recognizer.run_real_time_recognition()
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()