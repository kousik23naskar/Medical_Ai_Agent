from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal

from typing_extensions import TypedDict
from langgraph.types import Command
from src.agent_graph.pdf_rag_tool import query_pdf_chunks
from src.agent_graph.sql_tool import query_health_sqldb
from src.agent_graph.tavily_search_tool import query_tavily_web_search
from configs.load_tools_config import LoadToolsConfig
from src.utility import get_llm

# Load config
tool_cfg = LoadToolsConfig()

#Define state for the multi-agent system
class State(MessagesState):
    next: str

#Define router class for supervisor node
class Router(TypedDict):
    """
    Supervisor output schema.
    The 'next' field determines the next worker or whether the task is finished.
    Must be one of: RAG, SQL, websearch, chat, FINISH.
    """
    next: Literal["RAG", "SQL", "websearch", "chat", "FINISH"]

#Define all prompts for agents
rag_agent_prompt = """
You specialize in answering questions about GUIDELINES FOR INFECTION PREVENTION AND 
CONTROL using indexed PDF documents. Provide comprehensive, evidence-based answers.
"""

sql_agent_prompt = """
You can only answer queries related to Stroke_Prediction_Dataset, water_pollution_disease, 
survey_lung_cancer, Breast_Cancer tables from health_databse.db.
"""

# websearch_system_prompt = """
# You handle general health and medical queries through web search when specific databases 
# don't contain the answer. Focus on credible medical sources.
# """
websearch_agent_prompt = """
You are a web search expert that handles questions when other specialized agents 
(PDF-guideline agent or SQL medical database agent) cannot fully answer.

Use this tool only when:
- The user's question is unrelated to infection prevention or structured health datasets.
- Or if previous agents could not answer adequately.

Your domain covers:
- Health and medical questions (e.g., symptoms, causes, treatments, diagnosis)
- Weather and temperature in any city
- Sports news and scores
- Global and local news
- Politics and government
- History and major events
- Geography and country details
- Science, technology, discoveries
- Entertainment and pop culture
- Anything not covered by RAG or SQL agents

Your job is to find **up-to-date, accurate, and verifiable information** from trustworthy 
sources (e.g., scientific journals, health organizations, news outlets).

If you find nothing helpful online, return a clear response stating that no useful 
information was found.

Your answers must be well-structured and summarized for easy understanding.
"""

chat_agent_prompt = """
You are an intelligent assistant that handles general conversation, simple math, and code writing.

You do **not** rely on any external tools, databases, or documents.

You should handle:
- Greetings and small talk (e.g., "Hi", "What’s my name?", "How are you?")
- Personal or casual questions
- Simple to moderately complex math problems (e.g., "What is 22 * 8?")
- Code generation in Python or other common languages (e.g., "Write a Python script to reverse a list")

When generating code, format the response with proper markdown and explain the logic clearly.
Do not hallucinate tools — only use built-in capabilities.
If the question is unrelated to your scope, respond that another agent might be better suited.
"""

# ---------- NODES ----------

def rag_node(state: State, config: dict) -> Command[Literal["supervisor"]]:
    try:
        model_name = config["configurable"]["model_name"]
        llm = get_llm(model_name)
        rag_agent = create_react_agent(llm, tools=[query_pdf_chunks(model_name)], prompt=rag_agent_prompt)
        result = rag_agent.invoke(state)
        return Command(update={"messages": [AIMessage(content=result["messages"][-1].content, name="RAG")]}, goto="supervisor")        
    except Exception as e:
        return Command(update={"messages": [AIMessage(content=f"RAG agent error: {str(e)}", name="RAG")]}, goto="supervisor")

def sql_node(state: State, config: dict) -> Command[Literal["supervisor"]]:
    try:
        model_name = config["configurable"]["model_name"]
        llm = get_llm(model_name)
        sql_agent = create_react_agent(llm, tools=[query_health_sqldb(model_name)], prompt=sql_agent_prompt)

        # print("\n🧠 SQL agent state messages:") #debug lines
        # for m in state["messages"]:
        #     print(m)

        result = sql_agent.invoke(state)

        final_content = result["messages"][-1].content.strip()#debug lines
        print("🧾 SQL Agent result:", final_content) 

        return Command(update={"messages": [AIMessage(content=result["messages"][-1].content, name="SQL")]}, goto="supervisor")        
    except Exception as e:
        return Command(update={"messages": [AIMessage(content=f"SQL agent error: {str(e)}", name="SQL")]}, goto="supervisor")

def search_node(state: State, config: dict) -> Command[Literal["supervisor"]]:
    try:
        model_name = config["configurable"]["model_name"]
        llm = get_llm(model_name)
        search_agent = create_react_agent(llm, tools=[query_tavily_web_search], prompt=websearch_agent_prompt)        
        result = search_agent.invoke(state)
        return Command(update={"messages": [AIMessage(content=result["messages"][-1].content, name="websearch")]}, goto="supervisor")      
    except Exception as e:
        return Command(update={"messages": [AIMessage(content=f"Websearch agent error: {str(e)}", name="websearch")]}, goto="supervisor")
   
def chat_node(state: State, config: dict) -> Command[Literal["supervisor"]]:
    try:
        model_name = config["configurable"]["model_name"]
        llm = get_llm(model_name)
        chat_agent = create_react_agent(llm, tools=[], prompt=chat_agent_prompt)
        result = chat_agent.invoke(state)
        return Command(update={"messages": [AIMessage(content=result["messages"][-1].content, name="chat")]}, goto="supervisor")
    except Exception as e:
        return Command(update={"messages": [AIMessage(content=f"Chat agent error: {str(e)}", name="chat")]}, goto="supervisor")

# ---------- SUPERVISOR NODE ----------   
members = ["RAG", "SQL", "websearch", "chat"]
options = members + ["FINISH"]

#system prompt for supervisor node
system_prompt = f"""
You are a supervisor responsible for routing tasks to one of the following team members: {members} in a multi-agent system:

1. **RAG** — Specializes in answering questions about infection prevention and control guidelines from medical PDFs.
2. **SQL** — Answers questions that require querying structured medical data from tables such as:
   - **Stroke_Prediction_Dataset**: Features include patient demographics (age, gender), medical conditions (hypertension, heart disease), lifestyle (smoking), glucose and BMI levels, and stroke outcome.
   - **water_pollution_disease**: Includes water source types, contaminant levels (lead, nitrate, bacteria), water treatment methods, disease incidence (cholera, typhoid, diarrheal cases), and socio-economic/environmental factors (GDP, rainfall, population density).
   - **survey_lung_cancer**: Contains patient demographics (age, gender), smoking and alcohol habits, respiratory symptoms (coughing, wheezing, shortness of breath), anxiety, chronic disease, and lung cancer diagnosis.
   - **Breast_Cancer**: Contains clinical and demographic features such as tumor stage (T and N stages), tumor size, differentiation grade, hormone receptor status, lymph node involvement, survival months, and patient status (alive/dead).
3. **websearch** — Handles general or recent health questions not covered by the PDFs or SQL database.
4. **chat** — Handles casual conversation, personal queries (e.g. name or greetings), math questions, and basic code generation.
5. **FINISH** — If the question is fully answered, respond with FINISH.

Analyze user queries and route to the most appropriate agent. When responding, always provide complete and informative answers based on retrieved knowledge.
If no relevant information is found, state that clearly instead of returning a blank response.
Your output must always include a well-formed summary or explanation.

### IMPORTANT INSTRUCTIONS:

- You **must** return only a JSON object like: `{{ "next": "RAG" }}`.
- Valid values for `"next"` are: `"RAG"`, `"SQL"`, `"websearch"`, `"chat"`, or `"FINISH"`.

### Decision Guidelines:

1. If the question **is** related to **medical / health / pollution**, follow this flow:
   - First, route to `"RAG"` if it's about **infection prevention guidelines** (e.g. hygiene, pneumonia, antimicrobial resistance, etc), and it hasn't been used already.
   - Next, if the question involves **clinical or patient-level data, survival rates, disease stages, risk factors, or dataset-specific queries** (e.g. stroke prediction, water polution/pollution, Water Quality, waterborne diseases, hypertension, lung cancer, breast cancer, cancer stage), route to `"SQL"` if it hasn't yet been used.
   - Only use `"websearch"` if :
    - The question is medical/health-related,
    - AND neither `"RAG"` nor `"SQL"` have produced a complete or useful answer,
    - AND the information is likely not available in the internal PDFs or SQL database (e.g., recent research, global statistics).
2. If the user's question is clearly **not related to medical / health / pollution**, assign:
   - `"chat"` — for personal, math, or coding-related queries.
   - `"websearch"` — for general or current info (e.g., weather, news, history, geography, tech, sports, movies).
3. If the question is about **recent events** or **current affairs** (e.g., "What is the latest news on COVID-19?", "What are the current weather conditions in New York?"), assign: `"websearch"`.
4. If the question involves:
   - Personal or casual queries ("What is my name?", "Hello", "Tell me a joke")
   - Math operations ("What’s 15 x 8?", "Calculate 2^8")
   - Code writing ("Write a Python function to reverse a string")
   Then assign: `"chat"`   
5. Once an agent has responded during the current **query flow**, it should **not be re-invoked again** for the same question — even if its answer was incomplete.
6. If the previous agent’s answer was incomplete, try a **different agent** instead of repeating the same one.
7. Always consider both the **user’s query** and the **last agent’s response** to determine if FINISH is appropriate.
8. Do not return an empty or vague answer. Ensure the response is structured, informative, and user-friendly.
9. If an agent replies with phrases like "Sorry", "need more steps", "not enough data", or any other sign of incomplete capability, immediately route to another valid agent. Do not retry the same one.
10. If the most recent AI message contains a complete, well-structured answer (e.g. full explanation with bullet points, summary, etc.), return: `{{ "next": "FINISH" }}`.
"""


def supervisor_node(state: State, config: dict)-> Command[Literal[*members, "__end__"]]:
    try:
        model_name = config["configurable"]["model_name"]
        llm = get_llm(model_name)
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        # print("\n📨 Supervisor input messages:") #debug lines
        # for msg in messages:
        #     print(msg)

        response = llm.with_structured_output(Router).invoke(messages)
        #print("🧭 Supervisor routed to:", response) #debug line

        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})
    
    except Exception as e:
        print(f"Supervisor error: {e}")
        return Command(goto=END)
    
# ---------- GRAPH SETUP ----------
memory = MemorySaver()
builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("RAG", rag_node)
builder.add_node("SQL", sql_node)
builder.add_node("websearch", search_node)
builder.add_node("chat", chat_node)
graph = builder.compile(checkpointer=memory)

# ---------- GRAPH INVOCATION ----------
def custom_graph_invoke_output(user_question: str, model_name: str = "gpt-4o-mini"):
    """
    Invokes the graph and returns the final responding agent and its answer.

    Args:
        user_question (str): The user's question.
        model_name (str): LLM model (Ex. gpt, llama, mixtral).

    Returns:
        str: A clean, formatted response including the agent and final answer.
    """
    inputs = {
        "messages": [
            HumanMessage(content=user_question)
        ]
    }
    config = {
        "recursion_limit": 20,
        "configurable": {
            "thread_id": "chat_003",
            "model_name": model_name,
        }
    }
    try:
        result = graph.invoke(inputs, config=config)

        # Get the messages list
        #Example: result = {'messages': [AIMessage(content='...', name='RAG'), AIMessage(content='...', name='SQL')]}
        messages = result.get("messages", [])

        if messages and isinstance(messages, list):
            final_message = messages[-1]
            agent_name = getattr(final_message, "name", None)
            content = final_message.content.strip()

            if not content or content.lower() == user_question.lower():
                return "⚠️ No meaningful response generated. The agent may have echoed the input."
            
            if not agent_name:
                agent_name = "Unknown"

            return f"Agent: {agent_name}\nAnswer: {content}\n"
        else:
            return "⚠️ No meaningful response returned by any agent.\n"

    except Exception as e:
        return f"❌ Error during graph invocation: {str(e)}"