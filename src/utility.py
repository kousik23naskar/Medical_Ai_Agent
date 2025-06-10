from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from configs.load_tools_config import LoadToolsConfig

# Load config once
tool_cfg = LoadToolsConfig()

def get_llm(model_name: str, temperature: float = 0.0):
    if "gpt" in model_name:
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=tool_cfg.openai_api_key)
    elif "llama" in model_name or "mixtral" in model_name:
        return ChatGroq(model=model_name, temperature=temperature, api_key=tool_cfg.groq_api_key)
    else:
        raise ValueError(f"Unsupported model: {model_name}")