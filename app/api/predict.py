"""
Endpoint de predição de preços de imóveis.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.prediction import PredictionInput, PredictionOutput
from app.services.predictor import PredictorService, get_predictor_service
from utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post(
    "/",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Prediz o preço de um imóvel",
    description="Recebe características de um imóvel e retorna uma predição do preço",
)
async def predict(
    input_data: PredictionInput,
    predictor_service: PredictorService = Depends(get_predictor_service),
) -> PredictionOutput:
    """
    Endpoint para predição de preços de imóveis.

    Args:
        input_data: Dados de entrada do imóvel para predição.
        predictor_service: Serviço de predição injetado como dependência.

    Returns:
        PredictionOutput: Resultado da predição com o valor predito.

    Raises:
        HTTPException: Em caso de erro na predição.
    """
    try:
        log.info("Recebida solicitação de predição via endpoint /predict")
        result = predictor_service.predict(input_data)
        log.info("Predição realizada com sucesso pelo serviço")
        return result
    except ValueError as error:
        log.error("Erro ao carregar modelo: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao carregar modelo: {error}",
        ) from error
    except Exception as error:
        log.error("Erro inesperado durante predição: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao realizar predição: {error}",
        ) from error
