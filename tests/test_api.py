"""
Testes para os endpoints da API.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """
    Testa o endpoint raiz da API.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "status" in response.json()


def test_health_endpoint() -> None:
    """
    Testa o endpoint de health check.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_with_valid_input() -> None:
    """
    Testa o endpoint de predição com entrada válida.
    """
    # Arrange
    input_data = {
        "area": 100.0,
        "bedrooms": 3,
        "bathrooms": 2,
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "price" in data
    assert "confidence" in data
    assert isinstance(data["price"], (int, float))
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_with_invalid_input() -> None:
    """
    Testa o endpoint de predição com entrada inválida.
    """
    # Arrange
    input_data = {
        "area": -10.0,  # Valor inválido (deve ser > 0)
        "bedrooms": 3,
        "bathrooms": 2,
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_missing_fields() -> None:
    """
    Testa o endpoint de predição com campos faltando.
    """
    # Arrange
    input_data = {
        "area": 100.0,
        # Faltando bedrooms e bathrooms
    }

    # Act
    response = client.post("/predict/", json=input_data)

    # Assert
    assert response.status_code == 422  # Unprocessable Entity

