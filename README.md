# RAG Multi-Document Agent

Un agente RAG (Retrieval-Augmented Generation) para procesar y consultar múltiples documentos PDF.

## Características

- Procesamiento eficiente de documentos PDF
- Caché de embeddings para consultas rápidas
- Soporte para múltiples documentos
- Configuración flexible mediante archivo YAML
- Logging detallado de operaciones
- Manejo robusto de errores

## Requisitos

- Python 3.8+
- Poetry (para gestión de dependencias)
- API key de Mistral AI

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/rag-multidoc-agent.git
cd rag-multidoc-agent
```

2. Instalar dependencias:
```bash
poetry install
```

3. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env y agregar tu API key de Mistral AI
```

## Estructura del Proyecto

```
rag-multidoc-agent/
├── config.yaml           # Configuración del agente
├── src/
│   └── rag_multidoc_agent/
│       └── agent.py      # Clase principal del agente
├── examples/
│   └── use_rag_agent.py  # Ejemplo de uso
├── documents/            # Directorio para documentos PDF
└── data/
    └── embeddings/       # Caché de embeddings
```

## Uso

### Configuración

Edita el archivo `config.yaml` para ajustar:
- Modelos de LLM y embeddings
- Tamaño de chunks y overlap
- Rutas de directorios
- Configuración del motor de consultas

### Ejemplo Básico

```python
from rag_multidoc_agent.agent import RAGAgent

# Inicializar agente
agent = RAGAgent()

# Realizar consulta
response = agent.query(
    "documents/metagpt.pdf",
    "¿Cuáles son los puntos principales del documento?"
)
print(response)
```

### Forzar Recarga de Embeddings

```python
# Forzar recarga de embeddings (útil si el documento ha cambiado)
response = agent.query(
    "documents/metagpt.pdf",
    "¿Cuáles son los puntos principales del documento?",
    force_reload=True
)
```

## Mantenimiento de Embeddings

Los embeddings se guardan automáticamente en `data/embeddings/` con el formato:
```
{nombre_documento}_embeddings.pkl
```

Para limpiar el caché de embeddings:
```bash
rm -rf data/embeddings/*
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
