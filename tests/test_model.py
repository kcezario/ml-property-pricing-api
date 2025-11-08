"""
Testes para o modelo de Machine Learning e serviços relacionados.
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.config import settings
from app.schemas.prediction import PredictionInput, PredictionOutput
from app.services.predictor import PredictorService, reset_predictor_service_cache


def _build_valid_input() -> dict[str, float]:
    """
    Cria um dicionário com dados válidos para o schema de entrada.

    Returns:
        dict[str, float]: Dados simulados com base no dataset California Housing.
    """
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }


@pytest.fixture()
def mlflow_model_mock() -> Generator[MagicMock, None, None]:
    """
    Fixture que mocka chamadas ao MLflow durante os testes.

    Yields:
        MagicMock: Mock do modelo carregado.
    """
    with patch("app.services.predictor.mlflow.set_tracking_uri") as set_uri_mock, patch(
        "app.services.predictor.mlflow.pyfunc.load_model"
    ) as load_model_mock:
        model_mock = MagicMock()
        load_model_mock.return_value = model_mock
        yield model_mock
        set_uri_mock.assert_called_once_with(settings.MLFLOW_TRACKING_URI)
        load_model_mock.assert_called_once_with(
            f"models:/{settings.MODEL_NAME}@{settings.MODEL_STAGE}"
        )
        reset_predictor_service_cache()


def test_prediction_input_schema() -> None:
    """
    Testa a validação do schema de entrada.
    """
    input_data = PredictionInput(**_build_valid_input())

    for field_name, expected_value in _build_valid_input().items():
        assert getattr(input_data, field_name) == expected_value


def test_prediction_input_schema_invalid_value() -> None:
    """
    Garante que valores inválidos são rejeitados pelo schema de entrada.
    """
    invalid_data = _build_valid_input()
    invalid_data["AveRooms"] = -1.0

    with pytest.raises(Exception):
        PredictionInput(**invalid_data)


def test_prediction_output_schema() -> None:
    """
    Testa a validação do schema de saída.
    """
    output_data = PredictionOutput(predicted_value=4.526)
    assert output_data.predicted_value == 4.526


def test_predictor_service_initialization(mlflow_model_mock: MagicMock) -> None:
    """
    Testa a inicialização do serviço de predição.
    """
    service = PredictorService()

    assert service.model is mlflow_model_mock
    assert (
        service.model_uri
        == f"models:/{settings.MODEL_NAME}@{settings.MODEL_STAGE}"
    )


def test_predictor_service_predict(mlflow_model_mock: MagicMock) -> None:
    """
    Testa que o serviço de predição retorna valores esperados.
    """
    expected_value = 3.1415
    mlflow_model_mock.predict.return_value = [expected_value]

    service = PredictorService()
    input_data = PredictionInput(**_build_valid_input())

    result = service.predict(input_data)

    assert isinstance(result, PredictionOutput)
    assert result.predicted_value == expected_value

    mlflow_model_mock.predict.assert_called_once()
    called_df: pd.DataFrame = mlflow_model_mock.predict.call_args.args[0]
    assert list(called_df.columns) == settings.FEATURE_ORDER


def test_predictor_service_predict_without_model(mlflow_model_mock: MagicMock) -> None:
    """
    Testa que o serviço lança erro ao tentar predizer sem modelo carregado.
    """
    service = PredictorService()
    service.model = None
    input_data = PredictionInput(**_build_valid_input())

    with pytest.raises(ValueError, match="Modelo não foi carregado corretamente"):
        service.predict(input_data)

