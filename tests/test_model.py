"""
Testes para o modelo de Machine Learning e serviços relacionados.
"""

import pytest

from app.schemas.prediction import PredictionInput, PredictionOutput
from app.services.predictor import PredictorService


def test_prediction_input_schema() -> None:
    """
    Testa a validação do schema de entrada.
    """
    # Arrange & Act
    input_data = PredictionInput(area=100.0, bedrooms=3, bathrooms=2)

    # Assert
    assert input_data.area == 100.0
    assert input_data.bedrooms == 3
    assert input_data.bathrooms == 2


def test_prediction_input_schema_invalid_area() -> None:
    """
    Testa que o schema rejeita área inválida.
    """
    with pytest.raises(Exception):  # Pydantic ValidationError
        PredictionInput(area=-10.0, bedrooms=3, bathrooms=2)


def test_prediction_output_schema() -> None:
    """
    Testa a validação do schema de saída.
    """
    # Arrange & Act
    output_data = PredictionOutput(price=250000.0, confidence=0.92)

    # Assert
    assert output_data.price == 250000.0
    assert output_data.confidence == 0.92


def test_predictor_service_initialization() -> None:
    """
    Testa a inicialização do serviço de predição.
    """
    # Arrange & Act
    service = PredictorService()

    # Assert
    assert service.model is None
    assert service.model_name is None
    assert service.model_stage == "Production"


def test_predictor_service_predict_without_model() -> None:
    """
    Testa que o serviço lança erro ao tentar predizer sem modelo carregado.
    """
    # Arrange
    service = PredictorService()
    input_data = PredictionInput(area=100.0, bedrooms=3, bathrooms=2)

    # Act & Assert
    with pytest.raises(ValueError, match="Modelo não foi carregado"):
        service.predict(input_data)

