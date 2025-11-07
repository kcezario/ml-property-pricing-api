"""
Schemas Pydantic para entrada e saída de predições.
"""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """
    Schema de entrada para predição de preço de imóvel.

    Baseado no dataset California Housing, contém as seguintes features:
    - MedInc: Renda mediana do bloco
    - HouseAge: Idade mediana da casa
    - AveRooms: Número médio de quartos
    - AveBedrms: Número médio de quartos de dormir
    - Population: População do bloco
    - AveOccup: Ocupação média
    - Latitude: Latitude do bloco
    - Longitude: Longitude do bloco

    Attributes:
        MedInc: Renda mediana do bloco.
        HouseAge: Idade mediana da casa.
        AveRooms: Número médio de quartos.
        AveBedrms: Número médio de quartos de dormir.
        Population: População do bloco.
        AveOccup: Ocupação média.
        Latitude: Latitude do bloco.
        Longitude: Longitude do bloco.
    """

    MedInc: float = Field(..., description="Renda mediana do bloco")
    HouseAge: float = Field(..., description="Idade mediana da casa")
    AveRooms: float = Field(..., gt=0, description="Número médio de quartos")
    AveBedrms: float = Field(..., ge=0, description="Número médio de quartos de dormir")
    Population: float = Field(..., ge=0, description="População do bloco")
    AveOccup: float = Field(..., gt=0, description="Ocupação média")
    Latitude: float = Field(..., description="Latitude do bloco")
    Longitude: float = Field(..., description="Longitude do bloco")

    class Config:
        """Configuração do modelo Pydantic."""

        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }


class PredictionOutput(BaseModel):
    """
    Schema de saída para predição de preço de imóvel.

    Attributes:
        predicted_value: Valor predito do imóvel (em centenas de milhares de dólares).
    """

    predicted_value: float = Field(..., description="Valor predito do imóvel")

    class Config:
        """Configuração do modelo Pydantic."""

        json_schema_extra = {
            "example": {
                "predicted_value": 4.526,
            }
        }
