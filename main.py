from data_loader import DataLoader
from preprocessor import Preprocessor
from models import ModelTrainer
from evaluator import Evaluator

def main():
    """
    Fonction principale qui exécute tout le pipeline.
    """
    
    print("="*80)
    print("PROJET: CLASSIFICATION DE FEUILLES")
    print("="*80)
    
    # ========== 1. CHARGEMENT DES DONNÉES ==========
    print("\n[ÉTAPE 1/5] Chargement des données")
    print("-" * 80)
    
    data_loader = DataLoader('data/train.csv')
    data_loader.load_data()
    X, y = data_loader.split_features_target()
    
    # ========== 2. DIVISION TRAIN/VALIDATION ==========
    print("\n[ÉTAPE 2/5] Division des données")
    print("-" * 80)
    
    X_train, X_val, y_train, y_val = data_loader.create_train_val_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # ========== 3. PRÉTRAITEMENT ==========
    print("\n[ÉTAPE 3/5] Prétraitement des données")
    print("-" * 80)
    
    preprocessor = Preprocessor()
    
    # Encoder les labels (species -> nombres)
    y_train_encoded, y_val_encoded = preprocessor.encode_labels(y_train, y_val)
    
    # Standardiser les features
    X_train_scaled, X_val_scaled = preprocessor.scale_features(X_train, X_val)
    
    # ========== 4. ENTRAÎNEMENT DES MODÈLES ==========
    print("\n[ÉTAPE 4/5] Entraînement des modèles avec cross-validation")
    print("-" * 80)
    
    trainer = ModelTrainer(cv_folds=5, random_state=42)
    results = trainer.train_all_models(X_train_scaled, y_train_encoded)
    
    # ========== 5. ÉVALUATION ==========
    print("\n[ÉTAPE 5/5] Évaluation sur l'ensemble de validation")
    print("-" * 80)
    
    evaluator = Evaluator()
    df_results = evaluator.evaluate_all_models(results, X_val_scaled, y_val_encoded)
    
    # Rapport détaillé pour le meilleur modèle
    best_model_name = df_results.iloc[0]['model_name']
    best_model = results[best_model_name]['model']
    evaluator.detailed_report(
        best_model, 
        X_val_scaled, 
        y_val_encoded, 
        best_model_name,
        preprocessor.label_encoder
    )
    
    print("\n" + "="*80)
    print("PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("="*80)

if __name__ == "__main__":
    main()