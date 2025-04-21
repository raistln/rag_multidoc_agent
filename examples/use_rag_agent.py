import os
import sys
from pathlib import Path

# Agregar el directorio src al PYTHONPATH
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from rag_multidoc_agent.agent import RAGAgent

def main():
    # Inicializar el agente
    agent = RAGAgent()
    
    # Ruta al documento
    doc_path = Path("documents/metagpt.pdf")
    
    # Ejemplo de consulta
    question = "¿Cuáles son los puntos principales del documento?"
    
    try:
        # Realizar consulta
        response = agent.query(str(doc_path), question)
        print("\nRespuesta:")
        print(response)
        
        # Ejemplo de consulta forzando recarga de embeddings
        print("\nRealizando consulta con recarga de embeddings...")
        response = agent.query(str(doc_path), question, force_reload=True)
        print("\nRespuesta:")
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 