"""9-stage extraction orchestration shell (work/ artifacts only, no box writes).

Ported from B09 ``BookContentGenerator`` ``pipeline`` / ``observability`` /
``ingestion`` (read-only source) and bound to ``cbb_stage1.config`` /
``cbb_stage1.paths`` / ``cbb_stage1.models`` / ``cbb_stage1.engines.base.Engine``.
No Ollama, no FastAPI. Entry point: :func:`cbb_stage1.orchestrator.pipeline.run_pipeline`.
"""
