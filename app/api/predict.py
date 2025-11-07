"""
Endpoint de predição de preços de imóveis.
"""

from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import PredictionInput, PredictionOutput
from app.services.predictor import PredictorService

router = APIRouter(prefix="/predict", tags=["prediction"])

# Instância global do serviço de predição
predictor_service = PredictorService()


@router.post(
    "/",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Prediz o preço de um imóvel",
    description="Recebe características de um imóvel e retorna uma predição do preço",
)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Endpoint para predição de preços de imóveis.

    Args:
        input_data: Dados de entrada do imóvel para predição.

    Returns:
        PredictionOutput: Resultado da predição com o valor predito.

    Raises:
        HTTPException: Em caso de erro na predição.
    """
    try:
        result = predictor_service.predict(input_data)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao carregar modelo: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao realizar predição: {str(e)}",
        )
