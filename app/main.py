"""
Aplicação principal FastAPI.
"""

from fastapi import FastAPI

from app.api import predict

app = FastAPI(
    title="ML Property Pricing API",
    description="API para predição de preços de imóveis utilizando Machine Learning",
    version="0.1.0",
)

# Registrar rotas
app.include_router(predict.router)


@app.get("/", tags=["health"])
async def root():
    """
    Endpoint de health check.

    Returns:
        dict: Mensagem de status da API.
    """
    return {"message": "ML Property Pricing API", "status": "running"}


@app.get("/health", tags=["health"])
async def health():
    """
    Endpoint de health check detalhado.

    Returns:
        dict: Status detalhado da API.
    """
    return {"status": "ok"}

