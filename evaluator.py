from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

class Evaluator:
    """
    Classe pour évaluer les performances des modèles.
    """
    
    def evaluate_model(self, model, X_val, y_val, model_name):
        """
        Évalue un modèle sur l'ensemble de validation.
        """
        # Prédictions
        y_pred = model.predict(X_val)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return results
    
    def evaluate_all_models(self, models_dict, X_val, y_val):
        """
        Évalue tous les modèles et affiche un tableau comparatif.
        """
        print("\n" + "="*80)
        print("ÉVALUATION SUR L'ENSEMBLE DE VALIDATION")
        print("="*80)
        
        all_results = []
        
        for model_name, model_info in models_dict.items():
            model = model_info['model']
            cv_score = model_info['cv_score']
            
            results = self.evaluate_model(model, X_val, y_val, model_name)
            results['cv_score'] = cv_score
            all_results.append(results)
        
        # Création d'un DataFrame pour affichage
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values('accuracy', ascending=False)
        
        print("\nRésultats comparatifs:")
        print(df_results.to_string(index=False))
        
        # Meilleur modèle
        best_model_name = df_results.iloc[0]['model_name']
        best_accuracy = df_results.iloc[0]['accuracy']
        
        print(f"\n{'='*80}")
        print(f"MEILLEUR MODÈLE: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        print(f"{'='*80}")
        
        return df_results
    
    def detailed_report(self, model, X_val, y_val, model_name, label_encoder):
        """
        Affiche un rapport détaillé pour un modèle spécifique.
        """
        print(f"\n{'='*80}")
        print(f"RAPPORT DÉTAILLÉ: {model_name}")
        print(f"{'='*80}")
        
        y_pred = model.predict(X_val)
        
        # Rapport de classification
        print("\nRapport de classification:")
        target_names = label_encoder.classes_
        print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))