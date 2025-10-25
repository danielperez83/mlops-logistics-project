"""
Script de experimentación con diferentes hiperparámetros
Demuestra el valor de MLflow para tracking y comparación de experimentos
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
from datetime import datetime

from data_loader import load_config, load_data
from preprocessing import DataPreprocessor


def run_experiment(config, hyperparams, experiment_name):
    """
    Ejecuta un experimento con hiperparámetros específicos
    
    Args:
        config: Configuración base
        hyperparams: Diccionario con hiperparámetros a probar
        experiment_name: Nombre descriptivo del experimento
    """
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENTO: {experiment_name}")
    print(f"{'='*60}")
    print(f"Hiperparámetros: {hyperparams}")
    
    # Cargar y preprocesar datos
    df = load_data(config)
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess(df)
    
    # Entrenar modelo con hiperparámetros específicos
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Logging en MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=experiment_name):
        # Log hiperparámetros
        for param_name, param_value in hyperparams.items():
            mlflow.log_param(param_name, param_value)
        
        # Log métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log del modelo con signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:3]
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        # Tags para organización
        mlflow.set_tag("experiment_type", "hyperparameter_tuning")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Mostrar resultados
    print(f"\n📊 Resultados:")
    for metric_name, metric_value in metrics.items():
        print(f"   {metric_name}: {metric_value:.4f}")
    
    return metrics


def main():
    """
    Ejecuta múltiples experimentos con diferentes configuraciones
    """
    print("="*60)
    print("🔬 EXPERIMENTACIÓN CON HIPERPARÁMETROS")
    print("="*60)
    print("\nObjetivo: Encontrar la mejor configuración de hiperparámetros")
    print("usando MLflow para tracking y comparación\n")
    
    # Cargar configuración base
    config = load_config()
    
    # Definir diferentes configuraciones a probar
    experiments = [
        {
            "name": "baseline_small_trees",
            "params": {
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced"
            }
        },
        {
            "name": "baseline_medium_trees",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced"
            }
        },
        {
            "name": "deep_forest",
            "params": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced"
            }
        },
        {
            "name": "large_forest_shallow",
            "params": {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced"
            }
        },
        {
            "name": "balanced_config",
            "params": {
                "n_estimators": 150,
                "max_depth": 15,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced"
            }
        }
    ]
    
    # Ejecutar todos los experimentos
    results = []
    for exp in experiments:
        metrics = run_experiment(config, exp["params"], exp["name"])
        results.append({
            "name": exp["name"],
            **metrics
        })
    
    # Resumen comparativo
    print("\n" + "="*60)
    print("📈 RESUMEN COMPARATIVO DE EXPERIMENTOS")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Mejor modelo
    best_model = results_df.loc[results_df['f1_score'].idxmax()]
    print(f"\n🏆 MEJOR MODELO: {best_model['name']}")
    print(f"   F1-Score: {best_model['f1_score']:.4f}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   ROC-AUC: {best_model['roc_auc']:.4f}")
    
    print("\n" + "="*60)
    print("✅ EXPERIMENTACIÓN COMPLETADA")
    print("="*60)
    print("\nPara ver todos los experimentos en MLflow UI:")
    print("   mlflow ui")
    print("   http://localhost:5000")
    print("\nPuedes comparar:")
    print("   - Diferentes hiperparámetros")
    print("   - Métricas de cada modelo")
    print("   - Tiempos de entrenamiento")
    print("   - Seleccionar el mejor modelo para producción")


if __name__ == "__main__":
    main()