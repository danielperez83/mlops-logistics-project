"""
Módulo de preprocesamiento para el pipeline de ML
Maneja limpieza, codificación y escalado de datos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Clase para preprocesar datos del pipeline de logística
    """
    
    def __init__(self, config):
        """
        Inicializa el preprocesador con la configuración
        
        Args:
            config (dict): Configuración del proyecto
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess(self, df):
        """
        Ejecuta el pipeline completo de preprocesamiento
        
        Args:
            df (pd.DataFrame): Dataset original
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        print("\n🔧 Iniciando preprocesamiento...")
        
        # 1. Separar features y target
        X, y = self._split_features_target(df)
        
        # 2. Manejar valores nulos
        X = self._handle_missing_values(X)
        
        # 3. Codificar variables categóricas
        X = self._encode_categoricals(X)
        
        # 4. Escalar variables numéricas
        X_scaled = self._scale_features(X)
        
        # 5. Dividir en train y test
        X_train, X_test, y_train, y_test = self._split_train_test(
            X_scaled, y
        )
        
        print(f"✅ Preprocesamiento completado")
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def _split_features_target(self, df):
        """Separa features y variable objetivo"""
        target_col = self.config['data']['target_column']
        drop_cols = self.config['features']['drop']
        
        # Eliminar columnas no deseadas
        df_clean = df.drop(columns=drop_cols + [target_col])
        
        X = df_clean
        y = df[target_col]
        
        print(f"✅ Features: {X.shape[1]} columnas")
        print(f"✅ Target: {target_col}")
        
        return X, y
    
    def _handle_missing_values(self, X):
        """Maneja valores nulos"""
        # Contar nulos
        null_counts = X.isnull().sum()
        if null_counts.sum() > 0:
            print(f"⚠️  Valores nulos encontrados: {null_counts[null_counts > 0].to_dict()}")
            
            # Imputar numéricas con la mediana
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
            
            # Imputar categóricas con la moda
            cat_cols = X.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                X[cat_cols] = imputer.fit_transform(X[cat_cols])
            
            print("✅ Valores nulos imputados")
        else:
            print("✅ No hay valores nulos")
        
        return X
    
    def _encode_categoricals(self, X):
        """Codifica variables categóricas"""
        categorical_cols = self.config['features']['categorical']
        
        # Filtrar solo las columnas que existen en X
        categorical_cols = [col for col in categorical_cols if col in X.columns]
        
        if not categorical_cols:
            print("✅ No hay variables categóricas para codificar")
            return X
        
        print(f"🔄 Codificando {len(categorical_cols)} variables categóricas...")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"✅ Variables categóricas codificadas: {categorical_cols}")
        
        return X
    
    def _scale_features(self, X):
        """Escala las features numéricas"""
        print("📏 Escalando features...")
        
        # Guardar nombres de features
        self.feature_names = X.columns.tolist()
        
        # Escalar
        X_scaled = self.scaler.fit_transform(X)
        
        # Convertir de vuelta a DataFrame
        X_scaled = pd.DataFrame(
            X_scaled,
            columns=self.feature_names,
            index=X.index
        )
        
        print(f"✅ Features escaladas usando {self.config['preprocessing']['scaling_method']}")
        
        return X_scaled
    
    def _split_train_test(self, X, y):
        """Divide en train y test"""
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Mantener proporción de clases
        )
        
        print(f"✅ Datos divididos (test_size={test_size})")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test del módulo
    from data_loader import load_config, load_data
    
    config = load_config()
    df = load_data(config)
    
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess(df)
    
    print("\n✅ Módulo preprocessing.py funciona correctamente")
    print(f"   Features: {feature_names}")