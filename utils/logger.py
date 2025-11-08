"""
Ferramentas utilitárias para configuração de log da aplicação.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_DIR: Final[str] = "logs"


class CenteredLevelFormatter(logging.Formatter):
    """
    Formatter que centraliza o nome do nível de log.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Ajusta o nível do log para ficar centralizado.

        Args:
            record: Registro de log original.

        Returns:
            str: Mensagem formatada.
        """
        record.levelname = record.levelname.center(10)
        return super().format(record)


def _resolve_log_level() -> str:
    """
    Recupera o nível de log da aplicação.

    Returns:
        str: Nível de log em letras maiúsculas.
    """
    log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if log_level not in logging._nameToLevel:  # type: ignore[attr-defined]
        log_level = DEFAULT_LOG_LEVEL
    return log_level


def _resolve_log_dir() -> Path:
    """
    Recupera o diretório onde os logs serão escritos.

    Returns:
        Path: Caminho do diretório de logs.
    """
    log_dir = Path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger configurado com nome e nível padrão.

    Args:
        name: Nome do logger.

    Returns:
        logging.Logger: Instância configurada de logger.
    """
    logger = logging.getLogger(name)
    log_level = _resolve_log_level()
    log_dir = _resolve_log_dir()

    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = CenteredLevelFormatter("%(asctime)s [%(levelname)s] %(message)s")
        log_file = log_dir / f"{name.replace('.', '_')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
