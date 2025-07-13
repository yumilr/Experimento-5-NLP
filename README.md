# Experimento-5-NLP

## Estructura de carpetas y scripts

```
llama_lm_ppl/
├── datasets/
│   └── prepare_data.py
├── models/
│   └── load_model.py
├── evaluate/
│   └── compute_perplexity.py
├── run_experiment.py
└── requirements.txt
```

## Instrucciones de Compilación

```
pip install -r requirements.txt
python run_experiment.py
```

## Posibles percances

* Para que LLaMA-2 funcione necesitas aceptar su licencia en HuggingFace y estar autenticado.

* Si tu GPU no soporta FP16 o no tienes suficiente VRAM, usa la versión 8-bit (use_8bit=True) que reduce drásticamente el uso de memoria.

* Puedes cambiar fácilmente model_name a otro como EleutherAI/gpt-neo-1.3B para pruebas rápidas.

## Sobre bitnet

### ✅ Qué necesitas para que BitNet funcione

1. **Tener acceso a BitNet en Hugging Face** Si aún no tienes acceso, solicita el modelo desde su página oficial.

2. **Opcional: Usar 'transformers' en modo BF16 o FP16** Esto reduce el uso de memoria, aunque no es obligatorio para inferencia pura.

3. **Asegúrate que tu versión de 'transformers' ≥ 4.34** (Ya está cubierta en requirements.txt como 4.41.1)
