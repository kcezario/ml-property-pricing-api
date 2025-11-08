# API de Precificação de Imóveis com MLOps

[Click here for the English version of this README](README.en.md)

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white" alt="FastAPI Badge" />
  <img src="https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn Badge" />
  <img src="https://img.shields.io/badge/MLflow-2.8-0194E2?logo=mlflow&logoColor=white" alt="MLflow Badge" />
  <img src="https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white" alt="Docker Badge" />
  <img src="https://img.shields.io/badge/Pytest-Automated-0A9EDC?logo=pytest&logoColor=white" alt="Pytest Badge" />
  <img src="https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?logo=githubactions&logoColor=white" alt="GitHub Actions Badge" />
</p>

---

## 📌 Visão Geral do Projeto

Este projeto implementa um serviço de Machine Learning de ponta a ponta (end-to-end) para prever o preço de imóveis. Mais do que um simples modelo, esta é uma demonstração de um produto de ML robusto, seguindo as melhores práticas de MLOps, engenharia de software e automação.

![Demonstração da API](./utils/api_demo.gif)

## ✨ Principais Funcionalidades

- 🤖 **Pipeline de Treinamento Automatizado**: Treina, avalia e versiona um modelo de regressão usando Scikit-learn.
- 🔍 **Rastreabilidade com MLflow**: Registra experimentos, parâmetros, métricas e artefatos, garantindo reprodutibilidade.
- 🚀 **API de Inferência de Alta Performance**: Serve o modelo através de uma API RESTful assíncrona com FastAPI.
- 🐳 **Ambiente Containerizado**: O Docker garante que a aplicação rode de forma consistente em qualquer ambiente.
- ✅ **Qualidade de Código Garantida**: Testes automatizados com Pytest e linting com Ruff.
- 🔄 **CI/CD Automatizado**: Um workflow de GitHub Actions que valida o código a cada push, garantindo a integridade da base de código.

## 🧠 Arquitetura da Solução

O fluxo começa com o script de treinamento (`scripts/train.py`), que processa os dados do dataset California Housing e registra o modelo treinado no MLflow. A API FastAPI (`app/`) carrega o modelo mais recente marcado com o alias `staging` e oferece o endpoint `/predict` para inferências em tempo real. Todo o sistema é empacotado em uma imagem Docker para assegurar portabilidade e facilitar o deploy em diferentes ambientes.

## 🛠️ Stack de Tecnologia

- **Backend**
  - FastAPI, Uvicorn
- **Machine Learning**
  - Scikit-learn, Pandas, NumPy
- **MLOps**
  - MLflow (experimentos, registro de modelos)
  - Poetry (gestão de dependências)
- **Infraestrutura & DevOps**
  - Docker, GitHub Actions
- **Qualidade & Observabilidade**
  - Pytest, Ruff, Logging estruturado

## 🧪 Como Executar Localmente

1. **Clone o repositório**
   ```bash
   git clone https://github.com/<seu-usuario>/ml-property-pricing-api.git
   cd ml-property-pricing-api
   ```
2. **Instale o Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. **Instale as dependências**
   ```bash
   poetry install
   ```
4. **Inicie o servidor do MLflow**
   ```bash
   poetry run mlflow ui
   ```
5. **Execute o pipeline de treinamento**
   ```bash
   poetry run python scripts/train.py
   ```
6. **Promova o modelo para o alias `staging`**
   - Acesse o MLflow UI (por padrão em `http://127.0.0.1:5000`)
   - Localize o modelo treinado e defina o alias `staging`
7. **Inicie a API FastAPI**
   ```bash
   poetry run uvicorn app.main:app --reload
   ```
8. **Explore a documentação interativa**
   - Abra `http://127.0.0.1:8000/docs` no navegador

## 🧬 Executando os Testes

```bash
poetry run pytest
```
