"""
Aplicação principal FastAPI.
"""

from fastapi import FastAPI

from app.api import predict
from utils.logger import get_logger

log = get_logger(__name__)

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
    log.debug("Solicitação recebida no endpoint raiz")
    return {"message": "ML Property Pricing API", "status": "running"}


@app.get("/health", tags=["health"])
async def health():
    """
    Endpoint de health check detalhado.

    Returns:
        dict: Status detalhado da API.
    """
    log.debug("Verificação de saúde executada com sucesso")
    return {"status": "ok"}

