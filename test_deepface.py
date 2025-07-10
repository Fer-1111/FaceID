import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf

print("🚀 Probando DeepFace...")
print(f"TensorFlow version: {tf.__version__}")
print("✅ DeepFace importado correctamente")

# Test 1: Verificar que DeepFace funciona con enforce_detection=False
print("\n📝 Test 1: Verificar funcionamiento básico...")
try:
    # Crear imagen de prueba (no necesita ser una cara real)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
    print("✅ Test 1 exitoso: DeepFace funciona correctamente")
except Exception as e:
    print(f"❌ Test 1 falló: {e}")

# Test 2: Probar con la cámara web
print("\n📹 Test 2: Probando con cámara web...")
print("Presiona 'q' para salir del test")

try:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara")
    else:
        print("✅ Cámara iniciada correctamente")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo capturar frame")
                break
            
            frame = cv2.flip(frame, 1)  # Efecto espejo
            
            # Intentar análisis cada 30 frames para no sobrecargar
            if frame_count % 30 == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], 
                                            enforce_detection=False, silent=True)
                    
                    # Mostrar resultados en consola
                    if isinstance(result, list):
                        for i, face in enumerate(result):
                            print(f"🎭 Cara {i+1}: {face['dominant_emotion']} - {face['age']} años - {face['dominant_gender']}")
                    else:
                        print(f"🎭 Emoción: {result['dominant_emotion']} - {result['age']} años - {result['dominant_gender']}")
                    
                    # Mostrar texto en el frame
                    cv2.putText(frame, "DeepFace funcionando correctamente!", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"⚠️ Error en análisis: {e}")
                    cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Mostrar información básica
            cv2.putText(frame, "Test DeepFace - Presiona 'q' para salir", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Test DeepFace', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test 2 completado")

except Exception as e:
    print(f"❌ Error en test de cámara: {e}")

print("\n🎉 Pruebas completadas!")
print("Si viste emociones detectadas, ¡DeepFace está funcionando perfectamente!")
print("Ahora puedes ejecutar el código principal de reconocimiento facial.")