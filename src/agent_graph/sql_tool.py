import pandas as pd
from typing import List, Callable
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate#, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field


from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain.chains import create_sql_query_chain
from langchain.tools import tool
from configs.load_tools_config import LoadToolsConfig
from src.utility import get_llm

# Load config
tool_cfg = LoadToolsConfig()

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")


class HealthSQLAgent:
    """
    A specialized SQL agent that interacts with the Health SQL database using an LLM.

    Attributes:
        sql_agent_llm (ChatOpenAI/ChatGroq): The language model used.
        db (SQLDatabase): The connected SQL database.
        full_chain (Runnable): The complete pipeline from question to answer.
    """

    def __init__(self, sqldb_directory: str, llm, table_details_path: str) -> None:
        # LLM
        self.sql_agent_llm = llm

        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")

        self.table_details = self._get_table_details(table_details_path)

        # Step 1: Table extraction setup
        table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. 
        The tables are:

        {self.table_details}

        Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

        table_chain = create_extraction_chain_pydantic(
            Table, self.sql_agent_llm, system_message=table_details_prompt
        )
        #table_chain: [Table(name="Stroke_Prediction_Dataset"), Table(name="water_pollution_disease"), ...]

        
        self.select_table = (                            # Earlier used
            {"input": itemgetter("question")}
            | table_chain
            | self._get_tables
        )
        
        # # Create a ChatPromptTemplate with system and human messages
        # table_selection_prompt = ChatPromptTemplate.from_messages([
        #     SystemMessagePromptTemplate.from_template("{table_info}"),
        #     HumanMessagePromptTemplate.from_template("{question}")
        # ])

        # # Wrap the LLM with structured output for a list of tables
        # table_extractor = self.sql_agent_llm.with_structured_output(List[Table])

        # # Create a ChatPromptTemplate with system and human messages
        # table_selection_prompt = ChatPromptTemplate.from_messages([
        #     SystemMessagePromptTemplate.from_template("{table_info}"),
        #     HumanMessagePromptTemplate.from_template("{question}")
        # ])      
        
        # # Build the full table chain
        # self.select_table = (
        #     {"question": itemgetter("question"), "table_info": lambda _: table_details_prompt}
        #     | table_selection_prompt
        #     | table_extractor
        #     | self._get_tables
        # )

        # Step 2: SQL generation
        sql_db_query_prompt = PromptTemplate(
            input_variables=["input", "top_k", "table_info"],
            template="""
        You are a SQL assistant. Your task is to generate **safe, read-only SQL queries** only.

        Given the following table information: {table_info}
        The user wants at most {top_k} results.
        Answer the following question: {input}

        Constraints:
        - If the question has multiple parts, generate multiple queries separated by semicolons or a single SQL query.
        - Do NOT use INSERT, UPDATE, DELETE, DROP, or any other data-modifying operations.
        - Do NOT combine aggregation and non-aggregation in a single query without proper GROUP BY.

        Format:
        - Output only the SQL query statement(s)
        - Do NOT include dialect prefix (ex: "```sqlite", "```sql"), code block markers (ex: "```"), or explanations.

        Respond with clean and readable SQL query code (single or multiple lines) that can be safely executed on the database.
        """
        )

        self.generate_query = create_sql_query_chain(
            self.sql_agent_llm, self.db, prompt=sql_db_query_prompt
        )

        # Step 3: Query execution
        self.execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        self.multi_query_runner = RunnableLambda(
            lambda x: self._execute_multiple_sql_queries(x["query"], self.execute_query_tool)
        )

        # Step 4: Answer rephrasing
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer:"""
        )

        self.rephrase_answer = answer_prompt | self.sql_agent_llm | StrOutputParser()

        # Step 5: Combine into a chain
        self.full_chain = (
            RunnablePassthrough.assign(table_names_to_use=self.select_table)
            | RunnablePassthrough.assign(query=self.generate_query)
            .assign(result=self.multi_query_runner)
            | self.rephrase_answer
        )

    def _get_table_details(self, csv_path: str) -> str:
        """Reads CSV file and formats table name and description into a string."""
        df = pd.read_csv(csv_path)
        #df = pd.read_csv("database_table_descriptions.csv")
        table_details = ""
        for _, row in df.iterrows():
            table_details += f"Table Name: {row['Table']}\nTable Description: {row['Description']}\n\n"
        return table_details

    def _get_tables(self, tables: List[Table]) -> List[str]:
        """Extracts table names from Table model."""
        return [table.name for table in tables]

    def _execute_multiple_sql_queries(self, sql_code: str, db_tool) -> List:
        """Splits and executes multiple SQL queries."""
        queries = [q.strip() for q in sql_code.split(";") if q.strip()]        
        return [db_tool.invoke(q) for q in queries]

    def run(self, question: str, top_k: int = 5) -> str:
        """Executes the full chain on a user question."""
        return self.full_chain.invoke({"question": question, "top_k": top_k, "table_info": self.table_details})

## Final LangChain Tool wrapper
def query_health_sqldb(model_name: str) -> Callable:
    @tool
    def ask_health_sql(question: str) -> str:
        """
        Query the Health SQL Database using natural language.

        This tool allows users to ask health-related questions, which are automatically
        translated into safe, read-only SQL queries and executed on the health database.

        Available Tables:

        - Stroke_Prediction_Dataset  
        Description: Medical data for predicting stroke risk using patient features.

        - water_pollution_disease  
        Description: Global data on water pollution and related disease trends.

        - survey_lung_cancer  
        Description: Survey responses on lung cancer risk factors and symptoms.

        - Breast_Cancer  
        Description: Clinical data for analyzing breast cancer prognosis and outcomes.

        Args:
            question (str): A natural language question about the health data.
                        Example: "What factors most influence stroke risk in patients under 50?"

        Returns:
            str: A human-readable answer generated by executing the SQL query on the database.
        """
        try:
            # Initialize LLM based on model name
            llm = get_llm(model_name)
            # Initialize the SQL agent with config
            agent = HealthSQLAgent(
                sqldb_directory=tool_cfg.sql_db_path,
                llm=llm,
                table_details_path=tool_cfg.table_details_path,
            )

            # Run the full question → SQL → execution → final answer pipeline
            return agent.run(question)
        except Exception as e:
            return f"Error querying SQL health database: {str(e)}"
    return ask_health_sql