# Makefile para automatización del proyecto MLOps
# Proyecto: Smart Logistics Delay Prediction

.PHONY: help install train test lint clean all

# Variables
PYTHON := python3
PIP := pip
SRC_DIR := src
TEST_DIR := tests
VENV := .venv

# Ayuda por defecto
help:
	@echo "============================================"
	@echo "MLOps Logistics - Comandos Disponibles"
	@echo "============================================"
	@echo "make install      - Instalar dependencias"
	@echo "make train        - Entrenar modelo baseline"
	@echo "make experiment   - Ejecutar múltiples experimentos"
	@echo "make test         - Ejecutar tests"
	@echo "make lint         - Verificar código con flake8"
	@echo "make clean        - Limpiar archivos generados"
	@echo "make all          - Ejecutar pipeline completo"
	@echo "============================================"

# Instalar dependencias
install:
	@echo "📦 Instalando dependencias..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencias instaladas correctamente"

# Entrenar modelo (ejecutar pipeline completo)
train:
	@echo "🚀 Ejecutando pipeline de entrenamiento..."
	$(PYTHON) $(SRC_DIR)/train.py
	@echo "✅ Entrenamiento completado"

# Ejecutar experimentos con diferentes hiperparámetros
experiment:
	@echo "🔬 Ejecutando experimentos con múltiples configuraciones..."
	$(PYTHON) $(SRC_DIR)/experiment.py
	@echo "✅ Experimentación completada - Ver resultados en MLflow UI"

# Ejecutar tests
test:
	@echo "🧪 Ejecutando tests..."
	$(PYTHON) $(TEST_DIR)/test_basic.py
	@echo "✅ Tests completados"

# Linting con flake8
lint:
	@echo "🔍 Verificando código con flake8..."
	flake8 $(SRC_DIR) --max-line-length=100 --ignore=E501,W503,W293,W292,E226,F541,F401,W504
	@echo "✅ Linting completado"

# Limpiar archivos generados
clean:
	@echo "🧹 Limpiando archivos generados..."
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(TEST_DIR)/__pycache__
	rm -rf .pytest_cache
	rm -rf mlruns/
	rm -rf mlartifacts/
	rm -f feature_importance.csv
	rm -f *.log
	@echo "✅ Limpieza completada"

# Ejecutar todo el pipeline (install + lint + test + train)
all: install lint test train
	@echo "============================================"
	@echo "✅ PIPELINE COMPLETO EJECUTADO"
	@echo "============================================"