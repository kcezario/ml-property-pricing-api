"""
Script de treinamento do modelo de precificação de imóveis.
"""

import argparse
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados de treinamento.

    Args:
        file_path: Caminho para o arquivo de dados.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.
    """
    # TODO: Implementar carregamento de dados
    pass


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pré-processa os dados para treinamento.

    Args:
        df: DataFrame com os dados brutos.

    Returns:
        tuple: Tupla contendo features (X) e target (y).
    """
    # TODO: Implementar pré-processamento
    pass


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: Dict[str, Any],
) -> RandomForestRegressor:
    """
    Treina o modelo de Machine Learning.

    Args:
        X_train: Features de treinamento.
        y_train: Target de treinamento.
        hyperparameters: Hiperparâmetros do modelo.

    Returns:
        RandomForestRegressor: Modelo treinado.
    """
    # TODO: Implementar treinamento do modelo
    pass


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Avalia o modelo e retorna métricas.

    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Target de teste.

    Returns:
        dict: Dicionário com as métricas de avaliação.
    """
    # TODO: Implementar avaliação do modelo
    pass


def main() -> None:
    """
    Função principal do script de treinamento.
    """
    parser = argparse.ArgumentParser(description="Treina modelo de precificação de imóveis")
    parser.add_argument("--data-path", type=str, required=True, help="Caminho para os dados")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="property-pricing",
        help="Nome do experimento no MLflow",
    )
    args = parser.parse_args()

    # Configurar semente para reprodutibilidade
    np.random.seed(42)

    # Configurar experimento do MLflow
    mlflow.set_experiment(args.experiment_name)

    # TODO: Implementar fluxo completo de treinamento
    # with mlflow.start_run():
    #     # Carregar dados
    #     # Pré-processar
    #     # Treinar modelo
    #     # Avaliar modelo
    #     # Logar no MLflow
    #     pass


if __name__ == "__main__":
    main()

