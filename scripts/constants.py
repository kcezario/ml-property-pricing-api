"""
Constantes e configurações para o script de treinamento.
"""

from typing import Any

# Configuração dos argumentos do argparse
TRAIN_ARGUMENTS: dict[str, dict[str, Any]] = {
    "experiment-name": {
        "type": str,
        "default": "property-pricing",
        "help": "Nome do experimento no MLflow",
        "required": False,
    },
    "n-estimators": {
        "type": int,
        "default": 100,
        "help": "Número de árvores no RandomForest",
        "required": False,
    },
    "max-depth": {
        "type": int,
        "default": 10,
        "help": "Profundidade máxima das árvores",
        "required": False,
    },
    "test-size": {
        "type": float,
        "default": 0.2,
        "help": "Proporção dos dados para teste",
        "required": False,
    },
    "random-state": {
        "type": int,
        "default": 42,
        "help": "Semente aleatória para reprodutibilidade",
        "required": False,
    },
}

