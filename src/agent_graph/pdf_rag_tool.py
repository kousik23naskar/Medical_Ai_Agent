from operator import itemgetter
from typing import Callable
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from configs.load_tools_config import LoadToolsConfig
from src.utility import get_llm

# Load config
tool_cfg = LoadToolsConfig()

# Define prompt template
prompt = PromptTemplate.from_template(
    "Use the following medical context to answer the question.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)


def query_pdf_chunks(model_name: str) -> Callable:
    @tool
    def ask_pdf_guidelines(question: str) -> str:
        """
        Search the indexed medical "NATIONAL GUIDELINES FOR INFECTION PREVENTION AND CONTROL IN HEALTHCARE
        FACILITIES" PDFs using a semantic query and return raw matching content.

        This tool uses the existing Pinecone index of PDFs to find relevant chunks
        and returns them as-is (no formatting or generation). Intended for integration
        with a main agent that handles reasoning or prompting.

        Args:
            question (str): A natural language medical question.
                Example: "What are the guidelines for ventilator-associated pneumonia?"

        Returns:
            str: Joined raw document chunks with metadata for source PDF.
        """
        try:
            # Initialize LLM based on model name
            llm = get_llm(model_name)
            if not llm:
                return f"Unsupported model: {model_name}"
            
            # Pinecone client
            pc = Pinecone(api_key=tool_cfg.pinecone_api_key)
            index = pc.Index(tool_cfg.rag_pinecone_index)

            # Embedding and Vectorstore
            embeddings = HuggingFaceEmbeddings(model_name=tool_cfg.rag_embedding_model)
            vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
            #vectorstore = PineconeVectorStore(index_name=tool_cfg.rag_pinecone_index, embedding=embeddings)

            # Search top K chunks
            docs = vectorstore.similarity_search(question, k=tool_cfg.rag_k)

            if not docs:
                return "No matching content found."

            # Join and return raw text chunks with source info
            joined_docs = "\n\n".join(
                f"{doc.page_content.strip()}\n(Source: {doc.metadata.get('source_pdf', 'Unknown')})"
                for doc in docs
            )

            # Chain for final answer
            chain = (
                {"context": lambda _: joined_docs, "question": itemgetter("question")}
                | prompt
                | llm
                | StrOutputParser()
            )

            return chain.invoke({"question": question})
        except Exception as e:
            return f"Error querying PDF database: {str(e)}"
        
    return ask_pdf_guidelines # 🔁 ← this is the actual tool being passed back

#tools=[query_pdf_chunks(model_name)] is functionally identical to: tools=[ask_pdf_guidelines]  # already decorated
# you can even inspect the tool like
# tool = query_pdf_chunks("gpt-4o-mini")
# print(tool.name)         # "ask_pdf_guidelines"
# print(tool.description)  # From the docstring

#Callable: It returns something that is Callable — i.e., another function or callable object
#In other words, the return value of query_pdf_chunks(...) is another function that you can call
#Without return type also work properly i.e. def query_pdf_chunks(model_name: str):