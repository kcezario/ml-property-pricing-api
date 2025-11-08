"""
Serviço de predição de preços de imóveis.
"""

from functools import lru_cache

import mlflow
import mlflow.pyfunc
import pandas as pd

from app.config import settings
from app.schemas.prediction import PredictionInput, PredictionOutput
from utils.logger import get_logger

log = get_logger(__name__)


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
        log.info("Inicializando PredictorService")
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.model_uri = f"models:/{settings.MODEL_NAME}@{settings.MODEL_STAGE}"
        log.debug("Model URI configurada: %s", self.model_uri)
        try:
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            log.info("Modelo carregado com sucesso a partir do MLflow")
        except Exception as error:
            log.critical("Falha ao carregar modelo do MLflow: %s", error)
            raise

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
            log.error("Modelo não carregado ao tentar realizar predição")
            raise ValueError("Modelo não foi carregado corretamente.")

        log.debug("Convertendo dados de entrada para DataFrame")
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict], columns=settings.FEATURE_ORDER)

        log.debug("Iniciando predição com modelo carregado")
        prediction = self.model.predict(input_df)
        predicted_value = float(prediction[0])
        log.info("Predição concluída com sucesso")
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
