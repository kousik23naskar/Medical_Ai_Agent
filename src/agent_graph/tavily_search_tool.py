from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from configs.load_tools_config import LoadToolsConfig

# Load configuration
tool_cfg = LoadToolsConfig()

# Use the built-in Tavily search tool from LangChain
search_tool = TavilySearchResults(api_key=tool_cfg.tavily_api_key)

@tool
def query_tavily_web_search(query: str) -> str:
    """
    Perform a web search for general queries that are not answered by the RAG or SQL agent.

    Uses the Tavily Search API to retrieve the top relevant web pages and returns a concise 
    summary of their contents. This is ideal for questions related to:
    - Health and medicine (e.g., recent outbreaks, symptoms, treatments)
    - Weather and temperature
    - News and current events
    - History and politics
    - Entertainment and pop culture
    - Sports and technology
    - Other topics not covered by the infection prevention guideline PDF or structured SQL health database

    Args:
        query (str): A general, medical, or current-affairs question.
            Example: "What is the latest COVID-19 variant in 2025?"

    Returns:
        str: A concise summary of the top search results or a message if no useful info is found.
    """
    try:
        results = search_tool.invoke({"query": query, "num_results": tool_cfg.tavily_max_results})

        if not results or "results" not in results:
            return "No relevant web search results found."

        response = "\n\n".join(
            f"{r['title']}:\n{r['content']}\n(Source: {r['url']})"
            for r in results["results"]
        )

        return response
    except Exception as e:
        return f"Error performing web search: {str(e)}"
