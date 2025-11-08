"""
Testes para os endpoints da API.
"""

from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.schemas.prediction import PredictionInput, PredictionOutput
from app.services.predictor import PredictorService, get_predictor_service


@pytest.fixture
def predictor_service_mock() -> MagicMock:
    """
    Fixture que cria um mock para o PredictorService.

    Returns:
        MagicMock: Mock do serviço de predição.
    """
    return MagicMock(spec=PredictorService)


@pytest.fixture
def client(predictor_service_mock: MagicMock) -> Generator[TestClient, None, None]:
    """
    Fixture para criar uma instância do TestClient com serviço de predição mockado.

    Args:
        predictor_service_mock: Mock do serviço de predição.

    Yields:
        TestClient: Cliente de teste do FastAPI.
    """
    from app.main import app

    app.dependency_overrides[get_predictor_service] = lambda: predictor_service_mock

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_predictor_service, None)
    predictor_service_mock.reset_mock()




def test_root_endpoint(client: TestClient) -> None:
    """
    Testa o endpoint raiz da API.

    Args:
        client: Cliente de teste do FastAPI.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "status" in response.json()


def test_health_endpoint(client: TestClient) -> None:
    """
    Testa o endpoint de health check.

    Args:
        client: Cliente de teste do FastAPI.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_with_valid_input(
    client: TestClient, predictor_service_mock: MagicMock
) -> None:
    """
    Testa o endpoint de predição com entrada válida (Happy Path).

    Args:
        client: Cliente de teste do FastAPI.
    """
    # Arrange
    expected_predicted_value = 123.45
    mock_output = PredictionOutput(predicted_value=expected_predicted_value)
    predictor_service_mock.predict.return_value = mock_output

    input_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "predicted_value" in data
    assert data["predicted_value"] == expected_predicted_value
    predictor_service_mock.predict.assert_called_once()
    # Verificar que o argumento passado foi uma instância de PredictionInput
    call_args = predictor_service_mock.predict.call_args[0][0]
    assert isinstance(call_args, PredictionInput)


def test_predict_endpoint_with_invalid_input(
    client: TestClient, predictor_service_mock: MagicMock
) -> None:
    """
    Testa o endpoint de predição com entrada inválida (erro de validação).

    Args:
        client: Cliente de teste do FastAPI.
    """
    # Arrange - dados inválidos: AveRooms deve ser > 0
    input_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": -1.0,  # Valor inválido (deve ser > 0)
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 422  # Unprocessable Entity
    # Verificar que o serviço não foi chamado devido à validação do Pydantic
    predictor_service_mock.predict.assert_not_called()


def test_predict_endpoint_missing_fields(
    client: TestClient, predictor_service_mock: MagicMock
) -> None:
    """
    Testa o endpoint de predição com campos faltando.

    Args:
        client: Cliente de teste do FastAPI.
    """
    # Arrange - faltando campos obrigatórios
    input_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        # Faltando: AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 422  # Unprocessable Entity
    # Verificar que o serviço não foi chamado devido à validação do Pydantic
    predictor_service_mock.predict.assert_not_called()


def test_predict_endpoint_wrong_type(
    client: TestClient, predictor_service_mock: MagicMock
) -> None:
    """
    Testa o endpoint de predição com tipo de dado errado.

    Args:
        client: Cliente de teste do FastAPI.
    """
    # Arrange - tipo errado: MedInc deve ser float, não string
    input_data = {
        "MedInc": "not_a_number",  # Tipo inválido (deve ser float)
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 422  # Unprocessable Entity
    # Verificar que o serviço não foi chamado devido à validação do Pydantic
    predictor_service_mock.predict.assert_not_called()


def test_predict_endpoint_service_error(
    client: TestClient, predictor_service_mock: MagicMock
) -> None:
    """
    Testa o endpoint de predição quando o serviço lança uma exceção.

    Args:
        client: Cliente de teste do FastAPI.
    """
    # Arrange
    predictor_service_mock.predict.side_effect = Exception("Erro ao fazer predição")

    input_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 500  # Internal Server Error
    assert "Erro ao realizar predição" in response.json()["detail"]


def test_predict_endpoint_value_error(
    client: TestClient, predictor_service_mock: MagicMock
) -> None:
    """
    Testa o endpoint de predição quando o serviço lança um ValueError.

    Args:
        client: Cliente de teste do FastAPI.
    """
    # Arrange
    predictor_service_mock.predict.side_effect = ValueError(
        "Modelo não foi carregado corretamente"
    )

    input_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 500  # Internal Server Error
    assert "Erro ao carregar modelo" in response.json()["detail"]
