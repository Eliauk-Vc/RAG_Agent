"""Close request queues before flushing storage on service shutdown."""


async def finalize_rag(rag):
    try:
        for function in (rag.llm_model_func, getattr(rag.embedding_func, "func", None)):
            shutdown = getattr(function, "shutdown", None)
            if callable(shutdown):
                await shutdown()
    finally:
        await rag.finalize_storages()
