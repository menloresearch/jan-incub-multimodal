import logging
import time
from fastapi import FastAPI, Request
from .routers import transcription, models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Audio Transcription Service",
    description="OpenAI-compatible audio transcription API",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logging.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s"
    )
    response.headers["X-Process-Time"] = str(process_time)
    return response


app.include_router(transcription.router)
app.include_router(models.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
