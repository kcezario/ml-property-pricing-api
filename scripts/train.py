"""
Script de treinamento do modelo de precificação de imóveis.
"""

import argparse
from math import sqrt
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from constants import TRAIN_ARGUMENTS


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carrega o dataset California Housing do Scikit-learn.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Tupla contendo features (X) e target (y)
            como DataFrames/Series do Pandas.
    """
    # Carregar dataset
    california_housing = fetch_california_housing(as_frame=True)

    # Converter para DataFrames do Pandas
    X = california_housing.frame.drop(columns=["MedHouseVal"])
    y = california_housing.frame["MedHouseVal"]

    return X, y


def build_pipeline(
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> Pipeline:
    """
    Cria um pipeline de pré-processamento e modelagem.

    O pipeline contém:
    - StandardScaler: Para escalonar as features
    - RandomForestRegressor: Modelo de regressão

    Args:
        n_estimators: Número de árvores na floresta.
        max_depth: Profundidade máxima das árvores.
        random_state: Semente aleatória para reprodutibilidade.

    Returns:
        Pipeline: Pipeline do Scikit-learn com pré-processamento e modelo.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return pipeline


def main() -> None:
    """
    Função principal do script de treinamento.

    Orquestra todo o fluxo: carregamento de dados, divisão train/test,
    treinamento do pipeline, avaliação e registro no MLflow.
    """
    parser = argparse.ArgumentParser(description="Treina modelo de precificação de imóveis")

    # Adicionar argumentos a partir do dicionário de constantes
    for arg_name, arg_config in TRAIN_ARGUMENTS.items():
        parser.add_argument(
            f"--{arg_name}",
            type=arg_config["type"],
            default=arg_config["default"],
            help=arg_config["help"],
            required=arg_config.get("required", False),
        )

    args = parser.parse_args()

    # Configurar semente para reprodutibilidade
    np.random.seed(args.random_state)

    # Configurar experimento do MLflow
    mlflow.set_experiment(args.experiment_name)

    # Carregar dados
    print("Carregando dados...")
    X, y = load_data()
    print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")

    # Dividir dados em treino e teste
    print("Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    # Iniciar run do MLflow
    with mlflow.start_run():
        # Definir hiperparâmetros
        hyperparameters = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
        }

        # Log de parâmetros
        print("Registrando parâmetros no MLflow...")
        mlflow.log_params(hyperparameters)
        mlflow.log_param("test_size", args.test_size)

        # Criar pipeline
        print("Criando pipeline...")
        pipeline = build_pipeline(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

        # Treinar pipeline
        print("Treinando pipeline...")
        pipeline.fit(X_train, y_train)
        print("Pipeline treinado com sucesso!")

        # Fazer predições
        print("Fazendo predições no conjunto de teste...")
        y_pred = pipeline.predict(X_test)

        # Calcular métricas
        print("Calculando métricas...")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))

        metrics = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        }

        print(f"Métricas calculadas:")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        # Log de métricas
        print("Registrando métricas no MLflow...")
        mlflow.log_metrics(metrics)

        # Log de tags
        print("Adicionando tags no MLflow...")
        mlflow.set_tag("pipeline_description", "StandardScaler + RandomForest")
        mlflow.set_tag("dataset", "California Housing")

        # Log do modelo (pipeline completo)
        print("Registrando pipeline no MLflow...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="property-price-predictor",
            registered_model_name="property-price-predictor",
        )

        print("Modelo registrado com sucesso no MLflow!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
