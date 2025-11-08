"""
Serviço de predição de preços de imóveis.
"""

from functools import lru_cache

import mlflow
import mlflow.pyfunc
import pandas as pd

from app.config import settings
from app.schemas.prediction import PredictionInput, PredictionOutput


class PredictorService:
    """
    Serviço responsável por carregar o modelo e realizar predições.

    Attributes:
        model: Modelo de Machine Learning carregado do MLflow.
        model_uri: URI do modelo no MLflow Model Registry.
    """

    def __init__(self) -> None:
        """
        Inicializa o serviço de predição.

        Configura o tracking URI do MLflow e carrega o modelo do Model Registry.
        """
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.model_uri = f"models:/{settings.MODEL_NAME}@{settings.MODEL_STAGE}"
        self.model = mlflow.pyfunc.load_model(self.model_uri)

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Realiza a predição do preço do imóvel.

        Args:
            input_data: Dados de entrada do imóvel.

        Returns:
            PredictionOutput: Resultado da predição com o valor predito.

        Raises:
            ValueError: Se o modelo não estiver carregado.
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado corretamente.")

        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict], columns=settings.FEATURE_ORDER)
        prediction = self.model.predict(input_df)
        predicted_value = float(prediction[0])
        return PredictionOutput(predicted_value=predicted_value)


@lru_cache
def get_predictor_service() -> PredictorService:
    """
    Retorna uma instância singleton do PredictorService.

    Returns:
        PredictorService: Instância do serviço de predição.
    """
    return PredictorService()


def reset_predictor_service_cache() -> None:
    """
    Limpa o cache do serviço de predição.

    Útil para cenários de testes.
    """
    get_predictor_service.cache_clear()
