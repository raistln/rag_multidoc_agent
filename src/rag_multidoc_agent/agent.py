import os
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from langchain_mistralai import ChatMistralAI

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGAgent:
    """Agente RAG para procesamiento y consulta de documentos."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Inicializa el agente RAG con la configuración especificada.
        
        Args:
            config_path (str): Ruta al archivo de configuración YAML
        """
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Cargar variables de entorno
        load_dotenv()
        self.api_key = os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY no encontrada en variables de entorno")
        
        # Configurar modelos
        self._setup_models()
        
        # Crear directorios necesarios
        self._setup_directories()
        
        logger.info("Agente RAG inicializado correctamente")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga la configuración desde el archivo YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error al cargar configuración: {e}")
            raise
    
    def _setup_models(self):
        """Configura los modelos de LLM y embeddings."""
        try:
            # Configurar LLM
            Settings.llm = ChatMistralAI(
                model=self.config['model']['llm']['model']
            )
            
            # Configurar modelo de embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.config['model']['embedding']['model']
            )
            
            logger.info("Modelos configurados correctamente")
        except Exception as e:
            logger.error(f"Error al configurar modelos: {e}")
            raise
    
    def _setup_directories(self):
        """Crea los directorios necesarios si no existen."""
        try:
            for path in self.config['paths'].values():
                Path(path).mkdir(parents=True, exist_ok=True)
            logger.info("Directorios configurados correctamente")
        except Exception as e:
            logger.error(f"Error al configurar directorios: {e}")
            raise
    
    def save_embeddings(self, nodes: List[Any], doc_name: str) -> None:
        """Guarda los embeddings en un archivo.
        
        Args:
            nodes: Lista de nodos con embeddings
            doc_name: Nombre del documento
        """
        try:
            embeddings_path = Path(self.config['paths']['embeddings']) / f"{doc_name}_embeddings.pkl"
            with open(embeddings_path, 'wb') as f:
                pickle.dump(nodes, f)
            logger.info(f"Embeddings guardados para {doc_name}")
        except Exception as e:
            logger.error(f"Error al guardar embeddings: {e}")
            raise
    
    def load_embeddings(self, doc_name: str) -> Optional[List[Any]]:
        """Carga los embeddings desde un archivo.
        
        Args:
            doc_name: Nombre del documento
            
        Returns:
            Lista de nodos con embeddings o None si no existen
        """
        try:
            embeddings_path = Path(self.config['paths']['embeddings']) / f"{doc_name}_embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    nodes = pickle.load(f)
                logger.info(f"Embeddings cargados para {doc_name}")
                return nodes
            return None
        except Exception as e:
            logger.error(f"Error al cargar embeddings: {e}")
            raise
    
    def get_router_query_engine(self, file_path: str, force_reload: bool = False) -> RouterQueryEngine:
        """Crea un motor de consultas con soporte para caché de embeddings.
        
        Args:
            file_path: Ruta al archivo PDF
            force_reload: Si es True, recrea los embeddings aunque existan en caché
            
        Returns:
            RouterQueryEngine configurado
        """
        try:
            # Obtener nombre del documento
            doc_name = Path(file_path).stem
            
            # Intentar cargar embeddings desde caché
            nodes = None
            if not force_reload:
                nodes = self.load_embeddings(doc_name)
            
            # Si no hay embeddings en caché o se fuerza la recarga
            if nodes is None:
                logger.info(f"Procesando documento: {doc_name}")
                # Cargar documentos
                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                
                # Crear nodos
                splitter = SentenceSplitter(
                    chunk_size=self.config['processing']['chunk_size'],
                    chunk_overlap=self.config['processing']['chunk_overlap']
                )
                nodes = splitter.get_nodes_from_documents(documents)
                
                # Guardar embeddings para uso futuro
                self.save_embeddings(nodes, doc_name)
            
            # Crear índices
            summary_index = SummaryIndex(nodes)
            vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
            
            # Configurar motores de consulta
            summary_query_engine = summary_index.as_query_engine(
                response_mode=self.config['query_engine']['response_mode'],
                use_async=self.config['query_engine']['use_async'],
                llm=Settings.llm
            )
            vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
            
            # Crear herramientas de consulta
            summary_tool = QueryEngineTool.from_defaults(
                query_engine=summary_query_engine,
                description="Útil para preguntas de resumen relacionadas con el documento"
            )
            
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                description="Útil para recuperar contexto específico del documento"
            )
            
            # Crear motor de consultas con enrutamiento
            query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(),
                query_engine_tools=[summary_tool, vector_tool],
                verbose=self.config['query_engine']['verbose']
            )
            
            logger.info(f"Motor de consultas creado para {doc_name}")
            return query_engine
            
        except Exception as e:
            logger.error(f"Error al crear motor de consultas: {e}")
            raise
    
    def query(self, file_path: str, question: str, force_reload: bool = False) -> str:
        """Realiza una consulta sobre un documento.
        
        Args:
            file_path: Ruta al archivo PDF
            question: Pregunta a realizar
            force_reload: Si es True, recrea los embeddings
            
        Returns:
            Respuesta generada
        """
        try:
            query_engine = self.get_router_query_engine(file_path, force_reload)
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"Error al realizar consulta: {e}")
            raise 