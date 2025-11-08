# Imagem base
FROM python:3.12-slim

# Define o diretório de trabalho
WORKDIR /app

# Impede o Python de escrever arquivos .pyc e garante que a saída não seja bufferizada
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala o Poetry
RUN pip install --no-cache-dir poetry

# Copia os arquivos de dependência PRIMEIRO para aproveitar o cache do Docker
COPY pyproject.toml poetry.lock* ./

# Instala as dependências de produção NO AMBIENTE DO SISTEMA
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copia o código da aplicação
COPY ./app /app/app

# Expõe a porta que a aplicação vai rodar
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]