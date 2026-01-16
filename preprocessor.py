from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

class Preprocessor:
    """
    Classe pour prétraiter les données.
    Important: on fit seulement sur le train, puis on transform sur train ET validation.
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def encode_labels(self, y_train, y_val):
        """
        Transforme les noms de species (strings) en nombres.
        Exemple: 'Acer_Circinatum' -> 0
        """
        print("\nEncodage des labels...")
        
        # On fit sur le train
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        # On transform sur la validation
        y_val_encoded = self.label_encoder.transform(y_val)
        
        print(f"  Nombre de classes uniques: {len(self.label_encoder.classes_)}")
        
        return y_train_encoded, y_val_encoded
    
    def scale_features(self, X_train, X_val):
        """
        Standardise les features (moyenne=0, écart-type=1).
        Important pour les algorithmes sensibles à l'échelle (SVM, KNN, etc.)
        """
        print("Standardisation des features...")
        
        # On fit sur le train uniquement
        X_train_scaled = self.scaler.fit_transform(X_train)
        # On transform sur la validation avec les paramètres du train
        X_val_scaled = self.scaler.transform(X_val)
        
        print(f"  Moyenne des features (train): {np.mean(X_train_scaled):.4f}")
        print(f"  Std des features (train): {np.std(X_train_scaled):.4f}")
        
        return X_train_scaled, X_val_scaled