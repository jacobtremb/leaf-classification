from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    """
    Classe pour entraîner différents modèles avec recherche d'hyperparamètres.
    """
    
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        
    def get_models_and_params(self):
        """
        Définit les modèles à tester et leurs hyperparamètres.
        """
        models_params = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            
            'SVM': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga']
                }
            },
            
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            
            'MLP': {
                'model': MLPClassifier(random_state=self.random_state, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001]
                }
            }
        }
        
        return models_params
    
    def train_model(self, model_name, X_train, y_train):
        """
        Entraîne un modèle avec GridSearchCV pour la recherche d'hyperparamètres.
        """
        print(f"\n{'='*60}")
        print(f"Entraînement: {model_name}")
        print(f"{'='*60}")
        
        models_params = self.get_models_and_params()
        model_info = models_params[model_name]
        
        # GridSearchCV fait la cross-validation ET la recherche d'hyperparamètres
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Entraînement
        grid_search.fit(X_train, y_train)
        
        # Sauvegarde du meilleur modèle
        self.best_models[model_name] = grid_search.best_estimator_
        
        print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
        print(f"Score cross-validation: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_score_
    
    def train_all_models(self, X_train, y_train):
        """
        Entraîne tous les modèles définis.
        """
        print("\n" + "="*60)
        print("ENTRAÎNEMENT DE TOUS LES MODÈLES")
        print("="*60)
        
        results = {}
        models_params = self.get_models_and_params()
        
        for model_name in models_params.keys():
            model, cv_score = self.train_model(model_name, X_train, y_train)
            results[model_name] = {
                'model': model,
                'cv_score': cv_score
            }
        
        return results