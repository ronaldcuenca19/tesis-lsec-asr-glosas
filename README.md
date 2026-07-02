# tesis-lsec-asr-glosas
# Sistema de transcripción automática de voz en español y traducción a glosas LSEC

Este repositorio contiene los códigos, notebooks, configuraciones y recursos necesarios para reproducir el flujo desarrollado de voz a lengua de señas

El sistema implementa un flujo compuesto por:

1. Transcripción automática de voz en español mediante Whisper ajustado por fine-tuning.
2. Evaluación del modelo mediante Word Error Rate (WER).
3. Conversión de texto en español a glosas de Lengua de Señas Ecuatoriana.
4. Generación de una salida visual mediante concatenación de videos de señas.
5. Inferencia de todo el flujo de voz a LSEC mediante interfaz usando backend y frontend.

---

## Estructura del repositorio

```text
tesis-lsec-asr-glosas/
│
├── README.md
│
├── 01_entrenamiento_evaluacion_whisper/
│   ├── entrenamiento_evaluacion.ipynb
│   └── README.md
│
├── 02_traduccion_texto_LSEC/
│   ├── README.md
│   ├── traduccion_texto_LSEC.ipynb
│   ├── text_to_gloss.py
│   └── gloss_to_video.py
|
├── 03_inferencia/
│   ├── backend_inferencia/
|   |   ├── backend_inferencia.ipynb
|   |   └── README.md
|   └── frontend_inferencia/
│       └── README.md
|
