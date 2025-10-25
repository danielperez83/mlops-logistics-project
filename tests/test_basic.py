"""
Tests básicos para el pipeline de ML
"""

import pytest
import pandas as pd
import yaml
from pathlib import Path
import sys

# Agregar src/ al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_config, load_data, validate_data
from preprocessing import DataPreprocessor


def test_config_exists():
    """Test 1: Verificar que existe el archivo de configuración"""
    config_path = Path("config.yaml")
    assert config_path.exists(), "config.yaml no existe"
    print("✅ Test 1 pasado: config.yaml existe")


def test_load_config():
    """Test 2: Verificar que se puede cargar la configuración"""
    config = load_config()
    assert isinstance(config, dict), "La configuración debe ser un diccionario"
    assert 'data' in config, "Falta sección 'data' en config"
    assert 'model' in config, "Falta sección 'model' en config"
    assert 'mlflow' in config, "Falta sección 'mlflow' en config"
    print("✅ Test 2 pasado: Configuración cargada correctamente")


def test_dataset_exists():
    """Test 3: Verificar que existe el dataset"""
    config = load_config()
    data_path = Path(config['data']['raw_data_path'])
    assert data_path.exists(), f"Dataset no encontrado en {data_path}"
    print("✅ Test 3 pasado: Dataset existe")


def test_load_data():
    """Test 4: Verificar que se puede cargar el dataset"""
    config = load_config()
    df = load_data(config)
    
    assert isinstance(df, pd.DataFrame), "load_data debe retornar un DataFrame"
    assert not df.empty, "El DataFrame no debe estar vacío"
    assert len(df) > 0, "El DataFrame debe tener filas"
    
    print(f"✅ Test 4 pasado: Dataset cargado con {len(df)} filas")


def test_target_column_exists():
    """Test 5: Verificar que existe la columna objetivo"""
    config = load_config()
    df = load_data(config)
    target_col = config['data']['target_column']
    
    assert target_col in df.columns, f"Columna objetivo '{target_col}' no encontrada"
    print(f"✅ Test 5 pasado: Columna objetivo '{target_col}' existe")


def test_validate_data():
    """Test 6: Verificar que los datos son válidos"""
    config = load_config()
    df = load_data(config)
    
    is_valid = validate_data(df, config)
    assert is_valid, "Los datos no pasaron la validación"
    print("✅ Test 6 pasado: Datos validados correctamente")


def test_preprocessor_initialization():
    """Test 7: Verificar que el preprocesador se puede inicializar"""
    config = load_config()
    preprocessor = DataPreprocessor(config)
    
    assert preprocessor is not None, "No se pudo crear el preprocesador"
    assert preprocessor.config == config, "La configuración no se guardó correctamente"
    print("✅ Test 7 pasado: Preprocesador inicializado")


def test_preprocessing_pipeline():
    """Test 8: Verificar que el pipeline de preprocesamiento funciona"""
    config = load_config()
    df = load_data(config)
    preprocessor = DataPreprocessor(config)
    
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess(df)
    
    # Verificar shapes
    assert len(X_train) > 0, "X_train está vacío"
    assert len(X_test) > 0, "X_test está vacío"
    assert len(y_train) == len(X_train), "y_train no coincide con X_train"
    assert len(y_test) == len(X_test), "y_test no coincide con X_test"
    
    # Verificar que train es mayor que test
    assert len(X_train) > len(X_test), "Train debe ser mayor que test"
    
    # Verificar feature names
    assert len(feature_names) > 0, "No hay feature names"
    
    print(f"✅ Test 8 pasado: Preprocesamiento completado")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    # Ejecutar tests manualmente
    print("="*60)
    print("🧪 EJECUTANDO TESTS BÁSICOS")
    print("="*60 + "\n")
    
    tests = [
        test_config_exists,
        test_load_config,
        test_dataset_exists,
        test_load_data,
        test_target_column_exists,
        test_validate_data,
        test_preprocessor_initialization,
        test_preprocessing_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} falló: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 RESULTADOS: {passed} tests pasados, {failed} tests fallidos")
    print("="*60)
    
    if failed == 0:
        print("✅ TODOS LOS TESTS PASARON")
    else:
        print("⚠️  ALGUNOS TESTS FALLARON")