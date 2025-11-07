"""
Schemas Pydantic para entrada e saída de predições.
"""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """
    Schema de entrada para predição de preço de imóvel.

    Attributes:
        area: Área total do imóvel em metros quadrados.
        bedrooms: Número de quartos.
        bathrooms: Número de banheiros.
        # TODO: Adicionar mais campos conforme necessário
    """

    area: float = Field(..., gt=0, description="Área total do imóvel em m²")
    bedrooms: int = Field(..., ge=0, description="Número de quartos")
    bathrooms: int = Field(..., ge=0, description="Número de banheiros")

    class Config:
        """Configuração do modelo Pydantic."""

        json_schema_extra = {
            "example": {
                "area": 100.0,
                "bedrooms": 3,
                "bathrooms": 2,
            }
        }


class PredictionOutput(BaseModel):
    """
    Schema de saída para predição de preço de imóvel.

    Attributes:
        price: Preço predito do imóvel.
        confidence: Nível de confiança da predição (0.0 a 1.0).
    """

    price: float = Field(..., ge=0, description="Preço predito do imóvel")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Nível de confiança da predição"
    )

    class Config:
        """Configuração do modelo Pydantic."""

        json_schema_extra = {
            "example": {
                "price": 250000.0,
                "confidence": 0.92,
            }
        }

