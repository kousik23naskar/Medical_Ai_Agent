import os
import yaml
from pyprojroot import here
from dotenv import load_dotenv

# Load .env only once when config module is imported
load_dotenv()

class LoadToolsConfig:
    def __init__(self):
        with open(here("configs/tools_config.yaml")) as f:
            cfg = yaml.safe_load(f)

        # LLM Settings
        self.llm_models = cfg["llm_models"]
        self.default_llm = cfg["primary_agent"]["llm"]
        self.default_llm_temperature = float(cfg["primary_agent"]["llm_temperature"])

        # RAG (Pinecone-based)
        rag_cfg = cfg["guideipc_rag"]
        self.rag_embedding_model = rag_cfg["embedding_model"]
        self.rag_pinecone_index = rag_cfg["pinecone_index"]
        self.rag_k = int(rag_cfg["k"])

        # SQL DB
        self.sql_db_path = str(here(cfg["health_sqlagent_configs"]["health_sqldb_dir"]))
        self.table_details_path = str(here(cfg["health_sqlagent_configs"]["table_descriptions_file"]))

        # Web Search
        self.tavily_max_results = int(cfg["tavily_search_api"]["tavily_search_max_results"])

        # Whisper
        self.whisper_model = cfg["whisper_config"]["model"]
        self.whisper_provider = cfg["whisper_config"]["provider"]

        # PlayAI
        self.playai_model = cfg["playai_config"]["model"]
        self.playai_voice = cfg["playai_config"]["voice"]
        self.playai_response_format = cfg["playai_config"]["response_format"]

        # Graph
        self.thread_id = str(cfg["graph_configs"]["thread_id"])

        # Centralized API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        # Validate critical API keys
        missing_keys = []

        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.groq_api_key:
            missing_keys.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")
        if not self.pinecone_api_key:
            missing_keys.append("PINECONE_API_KEY")

        if missing_keys:
            raise ValueError(f"Missing required API keys in .env: {', '.join(missing_keys)}")   

        #Langchain project name
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT")
        if not self.langchain_project:
            raise ValueError("LANGCHAIN_PROJECT is not set in .env for LangSmith tracing.")
        
        # Enable LangSmith tracing (only if LANGCHAIN_TRACING_V2 is set to true)
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            if not os.getenv("LANGCHAIN_API_KEY"):
                raise ValueError("LANGCHAIN_API_KEY is not set in .env for LangSmith tracing.")
            
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            print(f"✅ LangSmith tracing enabled for project: {self.langchain_project}")