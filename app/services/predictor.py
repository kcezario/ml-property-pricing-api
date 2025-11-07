"""
Serviço de predição de preços de imóveis.
"""

from typing import Optional

import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator

from app.schemas.prediction import PredictionInput, PredictionOutput


class PredictorService:
    """
    Serviço responsável por carregar o modelo e realizar predições.

    Attributes:
        model: Modelo de Machine Learning carregado do MLflow.
    """

    def __init__(self, model_name: Optional[str] = None, model_stage: str = "Production"):
        """
        Inicializa o serviço de predição.

        Args:
            model_name: Nome do modelo no MLflow Model Registry.
            model_stage: Estágio do modelo (Production, Staging, etc.).
        """
        self.model: Optional[BaseEstimator] = None
        self.model_name = model_name
        self.model_stage = model_stage
        # TODO: Carregar modelo do MLflow quando implementado

    def load_model(self) -> None:
        """
        Carrega o modelo do MLflow Model Registry.

        Raises:
            ValueError: Se o modelo não puder ser carregado.
        """
        if self.model_name is None:
            # TODO: Implementar carregamento do modelo
            pass
        else:
            # TODO: Carregar modelo usando mlflow.pyfunc.load_model()
            # model_uri = f"models:/{self.model_name}/{self.model_stage}"
            # self.model = mlflow.pyfunc.load_model(model_uri)
            pass

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Realiza a predição do preço do imóvel.

        Args:
            input_data: Dados de entrada do imóvel.

        Returns:
            PredictionOutput: Resultado da predição.

        Raises:
            ValueError: Se o modelo não estiver carregado.
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado. Chame load_model() primeiro.")

        # TODO: Implementar pré-processamento dos dados
        # TODO: Chamar model.predict() com os dados pré-processados
        # TODO: Retornar PredictionOutput com o resultado

        # Placeholder
        return PredictionOutput(price=100000.0, confidence=0.95)


def get_predictor_service() -> PredictorService:
    """
    Factory function para criar instância do PredictorService.

    Returns:
        PredictorService: Instância do serviço de predição.
    """
    service = PredictorService()
    # TODO: Carregar modelo na inicialização quando implementado
    return service

