from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the AI Data Engineering Copilot",
    )


class Source(BaseModel):
    id: str
    section: str | None = None
    source: str | None = None


class AgentStepInfo(BaseModel):
    step_number: int
    action: str
    tool_name: str | None = None
    thought: str | None = None
    observation_summary: str | None = None
    latency_seconds: float = 0.0


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    latency_seconds: float
    cache_hit: bool = False
    request_id: str
    route: str = "rag"
    tool_name: str | None = None
    agent_steps: list[AgentStepInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
