# ML Property Pricing API

API para precificação de imóveis utilizando Machine Learning.

## Sobre o Projeto

Este projeto implementa uma API REST para predição de preços de imóveis utilizando modelos de Machine Learning treinados com Scikit-learn. A API é construída com FastAPI e utiliza MLflow para gerenciamento de modelos e experimentos.

## Tecnologias Utilizadas

- **FastAPI**: Framework web moderno e rápido para construção da API
- **Pydantic**: Validação de dados de entrada e saída
- **Scikit-learn**: Modelos de Machine Learning
- **Pandas**: Manipulação e análise de dados
- **MLflow**: Rastreamento de experimentos e versionamento de modelos
- **Pytest**: Framework de testes
- **Docker**: Containerização da aplicação
- **GitHub Actions**: CI/CD

## Como Executar Localmente

### Pré-requisitos

- Python 3.12 ou superior
- Poetry (gerenciador de dependências)

### Instalação

1. Clone o repositório:
```bash
git clone <repository-url>
cd ml-property-pricing-api
```

2. Instale as dependências usando Poetry:
```bash
poetry install
```

3. Ative o ambiente virtual:
```bash
poetry shell
```

4. Execute a API:
```bash
uvicorn app.main:app --reload
```

A API estará disponível em `http://localhost:8000`.

### Documentação da API

Após iniciar a API, a documentação interativa estará disponível em:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Estrutura do Projeto

```
/
├── app/                  # Módulo da API FastAPI
│   ├── __init__.py
│   ├── api/              # Endpoints/rotas
│   │   └── predict.py
│   ├── schemas/          # Modelos Pydantic (validação)
│   │   └── prediction.py
│   └── services/         # Lógica de negócio
│       └── predictor.py
├── notebooks/            # Jupyter notebooks para exploração
├── scripts/              # Scripts de treinamento
│   └── train.py
├── tests/                # Testes com Pytest
│   ├── test_api.py
│   └── test_model.py
├── .github/workflows/    # Workflows do GitHub Actions
│   └── ci.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Executando Testes

```bash
pytest
```

## Desenvolvimento

Este projeto segue os princípios de Clean Code e utiliza:
- **Black** para formatação de código
- **Ruff** para linting
- **Pytest** para testes
- **Conventional Commits** para mensagens de commit

