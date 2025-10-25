"""
Módulo de carga de datos para el pipeline de ML
Carga el dataset y realiza validaciones básicas
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Carga el archivo de configuración YAML
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        dict: Configuración del proyecto
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(config):
    """
    Carga el dataset desde la ruta especificada en la configuración
    
    Args:
        config (dict): Configuración del proyecto
        
    Returns:
        pd.DataFrame: Dataset cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el dataset está vacío
    """
    data_path = config['data']['raw_data_path']
    
    # Verificar que el archivo existe
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset no encontrado en: {data_path}")
    
    # Cargar datos
    print(f"📊 Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path)
    
    # Validaciones básicas
    if df.empty:
        raise ValueError("El dataset está vacío")
    
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Verificar que existe la columna objetivo
    target_col = config['data']['target_column']
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada")
    
    print(f"🎯 Variable objetivo: {target_col}")
    print(f"   Distribución: {df[target_col].value_counts().to_dict()}")
    
    return df


def validate_data(df, config):
    """
    Valida que el dataset tenga las columnas esperadas
    
    Args:
        df (pd.DataFrame): Dataset
        config (dict): Configuración del proyecto
        
    Returns:
        bool: True si la validación es exitosa
    """
    required_cols = (
        config['features']['numeric'] +
        config['features']['categorical'] +
        [config['data']['target_column']]
    )
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Columnas faltantes: {missing_cols}")
        return False
    
    print("✅ Todas las columnas requeridas están presentes")
    return True


if __name__ == "__main__":
    # Test del módulo
    config = load_config()
    df = load_data(config)
    validate_data(df, config)
    print("\n✅ Módulo data_loader.py funciona correctamente")