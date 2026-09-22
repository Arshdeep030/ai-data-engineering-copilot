from ollama import chat


MODEL = "qwen3:8b"


SYSTEM_PROMPT = """
You are the AI Data Engineering Copilot.

You help data engineers understand and solve
data engineering problems.

Your areas of expertise include:

- SQL
- Python
- Apache Spark
- Databricks
- Microsoft Fabric
- Apache Airflow
- Kafka
- dbt
- AWS
- Azure
- Data Warehousing
- ETL/ELT

Give technically accurate and practical explanations.
"""


def ask_llm(question: str) -> str:

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response["message"]["content"]