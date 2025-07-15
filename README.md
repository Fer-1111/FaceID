# Sistema de Reconocimiento Facial y Detección de Emociones

**Proyecto para Inteligencia Artificial**  

Sistema integral de reconocimiento facial y análisis de emociones en tiempo real con múltiples implementaciones que demuestran diferentes enfoques tecnológicos, desde algoritmos heurísticos hasta redes neuronales profundas.

---

## 🚀 Características Principales

### **Tres implementaciones especializadas:**
- **MediaPipe** - Detección rápida y eficiente con landmarks faciales
- **OpenCV + FER** - Balance óptimo entre precisión y rendimiento
- **DeepFace Integrado** - Sistema completo con reconocimiento de identidades

### **Funcionalidades avanzadas:**
- ✨ **Detección en tiempo real** - Reconoce caras y emociones instantáneamente
- 🎭 **7 emociones básicas** - Felicidad, tristeza, ira, miedo, sorpresa, disgusto, neutral
- 👤 **Registro de identidades** - Base de datos local de personas conocidas
- 🔄 **Modo dual** - Funciona con IA completa o modo básico de respaldo
- 📊 **Tracking temporal** - Seguimiento de patrones emocionales por persona
- 💾 **Persistencia de datos** - Almacenamiento automático de sesiones
- 🎯 **Sistema de calidad** - Métricas de confiabilidad en tiempo real

---

## 📋 Requisitos del Sistema

### **Software base:**
- Python 3.7+
- Cámara web
- Windows/Linux/macOS

### **Especificaciones recomendadas:**
| Implementación | RAM | CPU | GPU | Almacenamiento |
|---|---|---|---|---|
| **MediaPipe** | 4GB | Multi-core | Opcional | 500MB |
| **OpenCV + FER** | 8GB | Multi-core | Recomendada | 2GB |
| **DeepFace** | 16GB | Multi-core | Preferible | 5GB |

---

## 🛠️ Instalación

### **1. Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### **2. Crear entorno virtual (recomendado)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### **3. Instalar dependencias**

#### **Instalación básica (todas las implementaciones):**
```bash
pip install opencv-python numpy mediapipe fer tensorflow deepface
```

#### **O usar requirements.txt:**
```bash
pip install -r requirements.txt
```

#### **Instalación por módulos (opcional):**
```bash
# Solo MediaPipe
pip install opencv-python numpy mediapipe

# Solo OpenCV + FER
pip install opencv-python numpy fer tensorflow

# Solo DeepFace
pip install opencv-python numpy deepface tensorflow
```

### **4. Verificar instalación**
```bash
python test_deepface.py
```

---

## 📁 Estructura del Proyecto

```
facial-emotion-recognition/
├── 📄 mediaPipe.py              # Implementación MediaPipe
├── 📄 opencv_emotion.py         # Implementación OpenCV + FER
├── 📄 deepFace.py              # Implementación DeepFace integrada
├── 📄 reconocimiento_facial.py  # Sistema de reconocimiento básico
├── 📄 test_deepface.py          # Verificación de dependencias
├── 📄 requirements.txt          # Lista de dependencias
├── 📄 README.md                # Este archivo
├── 📁 mi_base_de_datos/        # Base de datos (se crea automáticamente)
│   ├── 📄 known_faces.json     # Información de identidades
│   ├── 📷 *.jpg               # Imágenes de referencia
│   └── 📊 session_*.json      # Datos de sesiones
├── 📁 emotion_data/           # Datos de emociones (MediaPipe/FER)
└── 📄 .gitignore             # Archivos ignorados
```

## 🎮 Guías de Uso

### **MediaPipe - Detección Rápida**
```bash
python mediaPipe.py
```
**Controles:**
- `l` - Activar/desactivar landmarks
- `f` - Mostrar/ocultar características
- `b` - Barras de confianza
- `r` - Reset estadísticas
- `s` - Guardar datos
- `1-7` - Calibración rápida de emociones
- `q` - Salir

### **OpenCV + FER - Precisión Equilibrada**
```bash
python opencv_emotion.py
```
**Controles:**
- `s` - Guardar sesión
- `i` - Información detallada
- `r` - Resetear estadísticas
- `+/-` - Ajustar velocidad de procesamiento
- `q` - Salir

### **DeepFace - Sistema Completo**
```bash
python deepFace.py
```
**Controles:**
- `r` - Registrar nueva persona
- `Espacio` - Capturar durante registro
- `i` - Información del sistema
- `s` - Guardar datos de sesión
- `h` - Mostrar/ocultar estadísticas
- `q` - Salir

### **Primer uso del sistema DeepFace:**
1. Ejecuta `python deepFace.py`
2. Presiona `r` para registrar
3. Ingresa el nombre de la persona
4. Posiciónate frente a la cámara
5. Presiona `Espacio` para capturar
6. ¡La persona queda registrada!

---

## ⚙️ Configuración Avanzada

### **Variables principales del sistema:**
```python
# En cada implementación puedes ajustar:
recognition_threshold = 0.6      # Umbral de confianza
frame_skip = 3                   # Frames a procesar
database_path = "mi_base_de_datos"  # Carpeta de datos
confidence_threshold = 0.3       # Confianza mínima
stabilization_frames = 8         # Frames para estabilización
```

### **Personalización de MediaPipe:**
```python
recognizer = ImprovedMediaPipeEmotionRecognition(save_data=True)
recognizer.process_every_n_frames = 2  # Más velocidad
recognizer.stabilization_frames = 15   # Más estabilidad
```

### **Personalización de FER:**
```python
recognizer = ImprovedEmotionRecognition(save_data=True)
recognizer.confidence_threshold = 0.5  # Más restrictivo
recognizer.min_face_size = (60, 60)    # Caras más grandes
```

### **Personalización de DeepFace:**
```python
recognizer = EnhancedFaceEmotionRecognition("mi_db")
recognizer.recognition_threshold = 0.7  # Más precisión
recognizer.frame_skip = 2              # Más velocidad
```

## 🐛 Solución de Problemas

### **DeepFace no se instala:**
```bash
# Instalar TensorFlow primero
pip install tensorflow==2.11.0
pip install deepface

# Si persiste el error
pip install --upgrade pip
pip install deepface --no-cache-dir
```

### **Error de cámara:**
- Verifica que la cámara esté conectada
- Permite acceso a la cámara en configuración del sistema
- Prueba cambiar el índice: `cv2.VideoCapture(1)` en lugar de `(0)`
- Cierra otras aplicaciones que usen la cámara

### **Problemas de rendimiento:**
```python
# Aumentar frame_skip para mejor FPS
frame_skip = 5  # Procesar cada 5 frames

# Reducir resolución
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Cerrar aplicaciones pesadas
```

### **MediaPipe no funciona:**
```bash
pip uninstall mediapipe
pip install mediapipe --no-cache-dir
```

### **FER da errores:**
```bash
pip install fer==22.5.1
pip install tensorflow==2.11.0
```

### **Errores de memoria:**
- Reduce `buffer_size` en las configuraciones
- Aumenta `frame_skip`
- Cierra otras aplicaciones
- Considera usar solo una implementación a la vez


### **Algoritmos implementados:**
- Redes Neuronales Convolucionales (CNN)
- Algoritmos heurísticos personalizados
- Análisis de landmarks faciales
- Métricas de distancia (coseno, euclidiana)
- Filtros de estabilización temporal

### **Versión actual (v1.0):**
- ✅ Tres implementaciones funcionales
- ✅ Reconocimiento facial básico
- ✅ Detección de 7 emociones
- ✅ Sistema de tracking
- ✅ Base de datos local

### **Documentación técnica:**
- [MediaPipe Documentation](https://mediapipe.dev/)
- [DeepFace GitHub](https://github.com/serengil/deepface)
- [OpenCV Documentation](https://opencv.org/)
- [TensorFlow Guides](https://tensorflow.org/)
