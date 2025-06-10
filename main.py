from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent_graph.multiagent_supervisor import custom_graph_invoke_output

# Initialize FastAPI app with a custom title
app = FastAPI(
    title="Medical AI Agent API",
    description="A multi-agent medical chatbot powered by LangGraph and OpenAI.",
    version="1.0.0"
)

# Define schema for query requests
class QueryRequest(BaseModel):
    question: str
    model_name: str

# Chat endpoint
@app.post("/chat/", summary="Query the Medical AI Agent")
async def chat_endpoint(request: QueryRequest):
    """
    Accepts a medical question and selected LLM model.
    Returns a response from the most suitable agent.
    """
    try:
        response = custom_graph_invoke_output(request.question, request.model_name)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Remove this block if you're using Docker's CMD to run uvicorn
# Run the app locally using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# To run this FastAPI app, use the command:
# uvicorn main:app --reload
# Swagger docs at http://127.0.0.1:8000/docs