"""
Endpoint de predição de preços de imóveis.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.prediction import PredictionInput, PredictionOutput
from app.services.predictor import PredictorService, get_predictor_service

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
        PredictionOutput: Resultado da predição com o preço estimado.

    Raises:
        HTTPException: Em caso de erro na predição.
    """
    try:
        # TODO: Implementar lógica de predição
        # Por enquanto, retorna um exemplo
        return PredictionOutput(price=100000.0, confidence=0.95)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao realizar predição: {str(e)}",
        )

