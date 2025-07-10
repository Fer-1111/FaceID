
Reconocimiento Facial en Tiempo Real
Sistema de reconocimiento facial en tiempo real usando OpenCV y DeepFace con capacidades de registro y detección automática.

🚀 Características

Detección en tiempo real - Reconoce caras instantáneamente desde la cámara
Registro de nuevas caras - Agrega personas a la base de datos fácilmente
Modo dual - Funciona con DeepFace (completo) o modo básico (solo detección)
Interfaz visual - Muestra nombres y confianza sobre las caras detectadas
Base de datos local - Almacena información de personas registradas
Rendimiento optimizado - Procesamiento asíncrono para mejor fluidez

📋 Requisitos

Python 3.7+
Cámara web
Windows/Linux/macOS

🛠️ Instalación

Clona el repositorio

bashgit clone https://github.com/tu-usuario/face-recognition-realtime.git
cd face-recognition-realtime

Instala las dependencias

bashpip install opencv-python numpy deepface
O usando requirements.txt:
bashpip install -r requirements.txt

Ejecuta el programa

bashpython face_recognition.py
🎮 Uso
Controles básicos:

'q' - Salir del programa
'r' - Registrar nueva cara (modo DeepFace)
'i' - Mostrar información del sistema
'Espacio' - Capturar foto durante registro

Primer uso:

Ejecuta el programa
Presiona 'r' para registrar una nueva cara
Ingresa el nombre de la persona
Posiciónate frente a la cámara
Presiona 'Espacio' para capturar
¡La persona queda registrada!

📁 Estructura del proyecto
face-recognition-realtime/
├── face_recognition.py      # Código principal
├── requirements.txt         # Dependencias
├── README.md               # Este archivo
├── mi_base_de_datos/       # Base de datos (se crea automáticamente)
│   ├── known_faces.json    # Información de caras registradas
│   └── *.jpg              # Imágenes de caras registradas
└── .gitignore             # Archivos ignorados por git
⚙️ Configuración
Variables principales:

recognition_threshold: Umbral de confianza (0.6 por defecto)
frame_skip: Frames a saltar para mejor rendimiento (3 por defecto)
database_path: Carpeta de la base de datos

Personalización:
python# En la clase RealTimeFaceRecognition
recognizer = RealTimeFaceRecognition(
    database_path="mi_base_de_datos",
    recognition_threshold=0.6,
    frame_skip=3
)
🔧 Modos de funcionamiento
Modo completo (con DeepFace):

Reconocimiento facial completo
Registro de nuevas caras
Análisis de confianza
Comparación con base de datos

Modo básico (sin DeepFace):

Solo detección de caras
Sin reconocimiento de identidad
Funciona como respaldo

🐛 Solución de problemas
DeepFace no se instala:
bashpip install tensorflow
pip install deepface
Cámara no detectada:

Verifica que la cámara esté conectada
Permite acceso a la cámara en configuración del sistema
Prueba cambiar el índice de cámara en el código

Problemas de rendimiento:

Aumenta frame_skip para mejor FPS
Reduce resolución de cámara
Cierra otras aplicaciones que usen la cámara

📊 Tecnologías utilizadas

OpenCV - Captura de video y detección de caras
DeepFace - Reconocimiento facial con IA
NumPy - Procesamiento de arrays
Threading - Procesamiento asíncrono
JSON - Almacenamiento de datos

🤝 Contribuir

Fork el proyecto
Crea una rama para tu feature (git checkout -b feature/nueva-caracteristica)
Commit tus cambios (git commit -m 'Agregar nueva característica')
Push a la rama (git push origin feature/nueva-caracteristica)
Abre un Pull Request

📝 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
🎯 Roadmap

 Reconocimiento de emociones
 Detección de edad y género
 Interfaz web con Flask
 Base de datos SQLite
 Reconocimiento múltiple simultáneo
 Exportar reportes
 Integración con APIs

👨‍💻 Autor
Fernando - GitHub
🙏 Agradecimientos

OpenCV community
DeepFace developers
Contribuidores del proyecto
