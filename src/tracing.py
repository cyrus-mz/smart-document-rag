"""Small LangSmith tracing decorators used at workflow boundaries."""

from langsmith import traceable


trace_ingest = traceable(name="pdf_ingestion", run_type="chain")
trace_chunking = traceable(name="extract_and_chunk_pdf", run_type="chain")
trace_retrieval = traceable(name="retrieve_chunks", run_type="retriever")
trace_node = traceable(run_type="chain")
trace_llm = traceable(name="ollama_completion", run_type="llm")
