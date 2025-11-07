"""
Configurações da aplicação usando Pydantic Settings.
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configurações da aplicação.

    Attributes:
        MODEL_NAME: Nome do modelo no MLflow Model Registry.
        MODEL_STAGE: Estágio do modelo (Production, Staging, etc.).
        MLFLOW_TRACKING_URI: URI do servidor de tracking do MLflow.
    """

    MODEL_NAME: str = "property-price-predictor"
    MODEL_STAGE: str = "staging"
    MLFLOW_TRACKING_URI: str = "mlruns"

    FEATURE_ORDER: List[str] = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    class Config:
        """Configuração do Pydantic Settings."""

        env_file = ".env"
        env_file_encoding = "utf-8"


# Instância global das configurações
settings = Settings()

