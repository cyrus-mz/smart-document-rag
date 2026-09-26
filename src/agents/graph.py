"""An explicit LangGraph with one possible rewrite and guaranteed termination."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import AgentModel, AgentState, RAGNodes, Retriever


def build_graph(retriever: Retriever, model: AgentModel):
    nodes = RAGNodes(retriever, model)
    graph = StateGraph(AgentState)
    graph.add_node("analyze_question", nodes.analyze_question)
    graph.add_node("retrieve_chunks", nodes.retrieve_chunks)
    graph.add_node("grade_evidence", nodes.grade_evidence)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate_answer", nodes.generate_answer)

    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "grade_evidence")

    def after_grade(state: AgentState) -> Literal["rewrite_query", "generate_answer"]:
        if state["evidence_good"] or state["rewrite_count"] >= 1:
            return "generate_answer"
        return "rewrite_query"

    graph.add_conditional_edges("grade_evidence", after_grade)
    graph.add_edge("rewrite_query", "retrieve_chunks")
    graph.add_edge("generate_answer", END)
    return graph.compile()
