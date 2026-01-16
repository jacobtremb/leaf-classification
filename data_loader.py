import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Classe pour charger et diviser les données.
    """
    
    def __init__(self, train_path):
        self.train_path = train_path
        self.data = None
        
    def load_data(self):
        """Charge les données depuis le fichier CSV"""
        print(f"Chargement des données depuis {self.train_path}...")
        self.data = pd.read_csv(self.train_path)
        print(f"Données chargées: {self.data.shape[0]} exemples, {self.data.shape[1]} colonnes")
        return self.data
    
    def split_features_target(self):
        """
        Sépare les features (X) et le target (y).
        Retire aussi l'id qui n'est pas utile pour la classification.
        """
        if self.data is None:
            self.load_data()
        
        # Les features sont toutes les colonnes sauf 'id' et 'species'
        X = self.data.drop(['id', 'species'], axis=1)
        # Le target est la colonne 'species'
        y = self.data['species']
        
        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"Nombre de classes: {y.nunique()}")
        
        return X, y
    
    def create_train_val_split(self, X, y, test_size=0.2, random_state=42):
        """
        Crée un split train/validation.
        Important: on ne touche pas aux données de test pour l'évaluation finale.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Garde la même proportion de classes
        )
        
        print(f"\nDivision train/validation:")
        print(f"  Train: {X_train.shape[0]} exemples")
        print(f"  Validation: {X_val.shape[0]} exemples")
        
        return X_train, X_val, y_train, y_val