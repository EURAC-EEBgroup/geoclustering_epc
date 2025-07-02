import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
import os
warnings.filterwarnings('ignore')
import optuna

class ModelloPredittivo:
    def __init__(self, file_path=None, target_column=None, problem_type='classificazione'):
        """
        Inizializzazione del modello predittivo
        
        Args:
            file_path: Percorso del file del dataset
            target_column: Nome della colonna target
            problem_type: Tipo di problema ('classificazione' o 'regressione')
        """
        self.file_path = file_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_score = float('-inf') if problem_type == 'classificazione' else float('inf')
        self.best_score_r2 = float('-inf') if problem_type == 'classificazione' else float('inf')
        
    def carica_dati(self, file_path=None, target_column=None):
        """
        Carica i dati da un file CSV o Excel
        
        Args:
            file_path: Percorso del file del dataset
            target_column: Nome della colonna target
        """
        if file_path:
            self.file_path = file_path
        if target_column:
            self.target_column = target_column
            
        if self.file_path:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError("Formato file non supportato. Utilizzare CSV o Excel.")
        else:
            # Usa un dataset di esempio se non è fornito un file
            from sklearn.datasets import load_iris, load_boston, fetch_california_housing
            if self.problem_type == 'classificazione':
                data = load_iris()
                self.df = pd.DataFrame(data=data.data, columns=data.feature_names)
                self.df['target'] = data.target
                self.target_column = 'target'
            else:  # regressione
                data = fetch_california_housing()
                self.df = pd.DataFrame(data=data.data, columns=data.feature_names)
                self.df['target'] = data.target
                self.target_column = 'target'
                
        print(f"Dataset caricato: {self.df.shape[0]} righe e {self.df.shape[1]} colonne")
        return self.df
        
    def analisi_esplorativa(self):
        """
        Esegue un'analisi esplorativa di base sui dati
        """
        if self.df is None:
            self.carica_dati()
            
        print("Informazioni sul dataset:")
        print(self.df.info())
        print("\nStatistiche descrittive:")
        print(self.df.describe())
        
        # Controlla valori mancanti
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nValori mancanti per colonna:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNessun valore mancante nel dataset.")
            
        # Visualizzazioni
        # if self.df.shape[1] <= 20:  # Limita la matrice di correlazione per dataset con molte features
        #     plt.figure(figsize=(10, 8))
        #     sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        #     plt.title('Matrice di Correlazione')
        #     plt.show()
        
        # Distribuzioni della variabile target
        # if self.target_column in self.df.columns:
        #     plt.figure(figsize=(8, 6))
        #     if self.problem_type == 'classificazione':
        #         self.df[self.target_column].value_counts().plot(kind='bar')
        #         plt.title('Distribuzione delle Classi')
        #     else:
        #         sns.histplot(self.df[self.target_column], kde=True)
        #         plt.title('Distribuzione della Variabile Target')
        #     plt.show()
        
        return self.df.head()
    
    def preprocess_data_prediction(self, df):   
        """
        Preprocessa i dati per la predizione
        """
        if self.df is None:
            self.carica_dati()
            
        if self.target_column not in self.df.columns:
            raise ValueError(f"La colonna target '{self.target_column}' non esiste nel dataset")
            
        # Separa feature e target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Identifica colonne categoriche e numeriche
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Definisci transformer per preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Applica preprocessing
        # X_processed = preprocessor.fit_transform(df) # da modficare con preprocessor.fit_transform(X)
        X_processed = X
                
        return X_processed
        

    def preprocess_dati(self, test_size=0.2, random_state=42):
        """
        Preprocessa i dati per l'addestramento
        
        Args:
            test_size: Proporzione del dataset da usare come test set
            random_state: Seed per la riproducibilità
        """
        if self.df is None:
            self.carica_dati()
            
        if self.target_column not in self.df.columns:
            raise ValueError(f"La colonna target '{self.target_column}' non esiste nel dataset")
            
        # Separa feature e target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Identifica colonne categoriche e numeriche
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Split train-test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Definisci transformer per preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Applica preprocessing
        # self.X_train_processed = preprocessor.fit_transform(self.X_train)
        # self.X_test_processed = preprocessor.transform(self.X_test)

        self.X_train_processed = self.X_train
        self.X_test_processed = self.X_test
        
        print(f"Dati preprocessati: {self.X_train_processed.shape[0]} campioni di training, {self.X_test_processed.shape[0]} campioni di test")
        print(f"Feature dopo preprocessing: {self.X_train_processed.shape[1]}")
        
        return self.X_train_processed, self.X_test_processed, self.y_train, self.y_test
    
    def addestra_modelli(self):
        """
        Addestra diversi modelli sul dataset
        """
        if self.X_train is None:
            self.preprocess_dati()
            
        # Definisci modelli in base al tipo di problema
        if self.problem_type == 'classificazione':
            models = {
                'RandomForest': RandomForestClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42),
                'LightGBM': lgb.LGBMClassifier(random_state=42)
            }
        else:  # regressione
            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'LightGBM': lgb.LGBMRegressor(random_state=42)
            }
        
        # Addestra e valuta ogni modello
        for name, model in models.items():
            print(f"\nAddestramento del modello: {name}")
            model.fit(self.X_train_processed, self.y_train)
            
            # Valutazione con cross-validation
            if self.problem_type == 'classificazione':
                cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, cv=5, scoring='accuracy')
                print(f"Accuracy in cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            else:
                try:
                    cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, cv=5, scoring='neg_root_mean_squared_error')
                    print(f"RMSE in cross-validation: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                except:
                    pass
            
            # Predizioni sul test set
            y_pred = model.predict(self.X_test_processed)
            
            # Valutazione delle performance
            if self.problem_type == 'classificazione':
                accuracy = accuracy_score(self.y_test, y_pred)
                print(f"Accuracy sul test set: {accuracy:.4f}")
                
                # Aggiorna il miglior modello
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_score_r2 = accuracy
                    self.best_model = name
            else:

                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                print(f"RMSE sul test set: {rmse:.4f}")
                print(f"MAE sul test set: {mae:.4f}")
                print(f"R² score: {r2:.4f}")
                
                # Aggiorna il miglior modello (basato su RMSE)
                if rmse < abs(self.best_score):
                    self.best_score = rmse
                    self.best_score_r2 = r2
                    self.best_model = name
            
            # Salva il modello nel dizionario
            self.models[name] = model
            
        print(f"\nMiglior modello: {self.best_model}")
        if self.problem_type == 'classificazione':
            print(f"Best score (accuracy): {self.best_score:.4f}")
        else:
            print(f"Best score (RMSE): {self.best_score:.4f}")
            
        return self.models
    
    def ottimizza_iperparametri(self, model_name=None):
        """
        Ottimizza gli iperparametri del modello specificato
        
        Args:
            model_name: Nome del modello da ottimizzare
        """
        if not model_name:
            model_name = self.best_model
            
        if not self.models:
            self.addestra_modelli()
            
        print(f"Ottimizzazione iperparametri per il modello: {model_name}")
        
        # Definisci parametri di ricerca in base al modello
        if model_name == 'RandomForest':
            if self.problem_type == 'classificazione':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
        elif model_name == 'XGBoost':
            if self.problem_type == 'classificazione':
                model = xgb.XGBClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            else:
                model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
        elif model_name == 'LightGBM':
            if self.problem_type == 'classificazione':
                model = lgb.LGBMClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 50, 70],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            else:
                model = lgb.LGBMRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 50, 70],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
        else:
            raise ValueError(f"Modello '{model_name}' non supportato")
            
        # Definisci la metrica di scoring appropriata
        if self.problem_type == 'classificazione':
            scoring = 'accuracy'
        else:
            scoring = 'neg_root_mean_squared_error'
            
        # Esegui GridSearchCV
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train_processed, self.y_train)
        
        # Estrai risultati
        print(f"Migliori parametri: {grid_search.best_params_}")
        print(f"Miglior score: {grid_search.best_score_:.4f}")
        
        # Aggiorna il modello con i migliori parametri
        best_model = grid_search.best_estimator_
        self.models[model_name] = best_model
        
        # Valuta il modello ottimizzato
        y_pred = best_model.predict(self.X_test_processed)
        
        if self.problem_type == 'classificazione':
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\nAccuracy del modello ottimizzato sul test set: {accuracy:.4f}")
            
            # Report di classificazione dettagliato
            print("\nReport di classificazione:")
            print(classification_report(self.y_test, y_pred))
            
            # Confusion matrix
            # plt.figure(figsize=(8, 6))
            # cm = confusion_matrix(self.y_test, y_pred)
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            # plt.title('Matrice di Confusione')
            # plt.xlabel('Predizioni')
            # plt.ylabel('Valori Reali')
            # plt.show()
            
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model_name + "_optimized"
        else:
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            print(f"\nRMSE del modello ottimizzato sul test set: {rmse:.4f}")
            print(f"MAE del modello ottimizzato sul test set: {mae:.4f}")
            print(f"R² score del modello ottimizzato: {r2:.4f}")
            
            # Visualizza predizioni vs valori reali
            # plt.figure(figsize=(10, 6))
            # plt.scatter(self.y_test, y_pred, alpha=0.5)
            # plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
            # plt.xlabel('Valori Reali')
            # plt.ylabel('Predizioni')
            # plt.title('Predizioni vs Valori Reali')
            # plt.show()
            
            if rmse < abs(self.best_score):
                self.best_score = rmse
                self.best_score_r2 = r2
                self.best_model = model_name + "_optimized"
                
        return best_model
    
    def feature_importance(self, model_name=None):
        """
        Visualizza l'importanza delle feature per il modello specificato
        
        Args:
            model_name: Nome del modello da analizzare
        """
        if not model_name:
            model_name = self.best_model
            if "_optimized" in model_name:
                model_name = model_name.replace("_optimized", "")
                
        if not self.models:
            self.addestra_modelli()
            
        model = self.models[model_name]
        
        # Verifica che il modello abbia un attributo per l'importanza delle feature
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Se abbiamo un ColumnTransformer, dobbiamo recuperare i nomi delle feature
            if isinstance(self.X_train, pd.DataFrame):
                feature_names = self.X_train.columns
            else:
                feature_names = [f"Feature {i}" for i in range(len(importances))]
                
            # Crea un DataFrame con le importanze
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Visualizza le top 20 feature (o meno se ci sono meno feature)
            n_features = min(20, len(feature_importance_df))
            # plt.figure(figsize=(10, 8))
            # sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(n_features))
            # plt.title(f'Top {n_features} Feature Importance - {model_name}')
            # plt.tight_layout()
            # plt.show()
            
            return feature_importance_df
        else:
            print(f"Il modello {model_name} non supporta la visualizzazione dell'importanza delle feature")
            return None
    
    def salva_modello(self, model, file_path=None, save_preprocessor=True):
        """
        Salva il modello specificato e opzionalmente il preprocessor su file
        
        Args:
            model_name: Nome del modello da salvare
            file_path: Percorso dove salvare il modello (senza estensione)
            save_preprocessor: Se salvare anche il preprocessor
        
        Returns:
            dict: Dizionario con i percorsi dei file salvati
        """
        import joblib
        import os
        
        model_name=self.best_model
        # if not model_name:
        #     model_name = self.best_model
            
        # if not self.models:
        #     self.addestra_modelli()
            
        # if not file_path:
        #     # Crea una directory models se non esiste
        #     if not os.path.exists('models'):
        #         os.makedirs('models')
        #     file_path = f"models/{model_name}"
        
        # # Assicurati che la directory esista
        # output_dir = os.path.dirname(file_path)
        # if output_dir and not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
            
        # model = self.models[model_name]
        
        # Salva il modello
        model_path = f"{file_path}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Modello {model_name} salvato come {model_path}")
        
        saved_files = {'model': model_path}
        
        # Salva anche il preprocessor se richiesto
        if save_preprocessor and hasattr(self, 'preprocessor'):
            preprocessor_path = f"{file_path}_preprocessor.joblib"
            joblib.dump(self.preprocessor, preprocessor_path)
            print(f"Preprocessor salvato come {preprocessor_path}")
            saved_files['preprocessor'] = preprocessor_path
        
        # Salva i metadati del modello
        import json
        metadata = {
            'model_name': model_name,
            'problem_type': self.problem_type,
            'target_column': self.target_column,
            'feature_columns': list(self.X_train.columns),
            'performance': {
                'metric': 'accuracy' if self.problem_type == 'classificazione' else 'rmse',
                'value': float(self.best_score),
                'metric2': 'accuracy' if self.problem_type == 'classificazione' else 'r2',
                'value2': float(self.best_score_r2)
            },
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = f"{file_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        saved_files['metadata'] = metadata_path
        print(f"Metadati del modello salvati come {metadata_path}")
    
        return saved_files

    def carica_modello(self, model_path, preprocessor_path=None):
        """
        Carica un modello salvato in precedenza
        
        Args:
            model_path: Percorso del file del modello
            preprocessor_path: Percorso del file del preprocessor
        
        Returns:
            model: Il modello caricato
        """
        import joblib
        
        # Carica il modello
        model = joblib.load(model_path)
        print(f"Modello caricato da {model_path}")
        
        # Carica anche il preprocessor se fornito
        if preprocessor_path:
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor caricato da {preprocessor_path}")
        
        # Aggiungi il modello al dizionario dei modelli
        model_name = os.path.basename(model_path).replace('_model.joblib', '')
        self.models[model_name] = model
        
        return model

    def predici(self, dati_input, best_model, apply_preprocessing=True, intervallo_confidenza=0.95):
        """
        Effettua predizioni su nuovi dati con intervallo di confidenza
        
        Args:
            dati_input: DataFrame o array con i dati su cui effettuare predizioni
            best_model: Modello da utilizzare per la predizione
            apply_preprocessing: Se applicare il preprocessor ai dati
            intervallo_confidenza: Livello di confidenza per l'intervallo (default: 0.95 per 95%)
            
        Returns:
            dict: Predizioni del modello con intervalli di confidenza
        """
        model = best_model
        
        # Prepara i dati input
        if isinstance(dati_input, pd.DataFrame):
            # Verifica che tutte le colonne necessarie siano presenti
            if hasattr(self, 'X_train') and isinstance(self.X_train, pd.DataFrame):
                missing_cols = set(self.X_train.columns) - set(dati_input.columns)
                if missing_cols:
                    raise ValueError(f"I dati di input mancano delle seguenti colonne: {missing_cols}")
            
            # Applica lo stesso preprocessing usato per i dati di training se richiesto
            if apply_preprocessing and hasattr(self, 'preprocessor'):
                print("Applicazione preprocessing ai dati...")
                dati_processati = self.preprocessor.transform(dati_input)
            else:
                # Se non è richiesto preprocessing o non abbiamo un preprocessor, 
                # assumiamo che i dati siano già formattati correttamente
                dati_processati = dati_input
        else:
            # Assumiamo che i dati siano già preprocessati correttamente
            dati_processati = dati_input
        
        # Effettua le predizioni
        predictions = model.predict(dati_processati)
        
        result = {'predictions': predictions}
        
        # Per i problemi di classificazione, offriamo anche le probabilità se il modello le supporta
        if self.problem_type == 'classificazione' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(dati_processati)
            result['probabilities'] = probabilities
        
        # Per i problemi di regressione, calcoliamo gli intervalli di confidenza
        if self.problem_type == 'regressione':
            # Per modelli che supportano il calcolo dell'incertezza direttamente (come alcune versioni di Random Forest)
            if hasattr(model, 'estimators_'):
                # Facciamo predizioni con tutti gli estimatori dell'ensemble
                preds_per_estimator = np.array([tree.predict(dati_processati) for tree in model.estimators_])
                
                # Calcoliamo media e deviazione standard delle predizioni
                mean_prediction = np.mean(preds_per_estimator, axis=0)
                std_prediction = np.std(preds_per_estimator, axis=0)
                
                # Calcoliamo l'intervallo di confidenza usando la distribuzione normale
                import scipy.stats as st
                alpha = 1 - intervallo_confidenza
                z_score = st.norm.ppf(1 - alpha/2)  # Two-tailed z-score
                
                ci_lower = mean_prediction - z_score * std_prediction
                ci_upper = mean_prediction + z_score * std_prediction
                
                result['ci_lower'] = ci_lower
                result['ci_upper'] = ci_upper
                
            # Per modelli XGBoost e LightGBM che supportano quantile regression
            elif isinstance(model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
                # Per XGBoost/LightGBM utilizziamo una stima basata sull'errore di predizione sul training set
                if hasattr(self, 'y_train') and hasattr(self, 'X_train_processed'):
                    # Calcoliamo l'errore sul training set
                    y_train_pred = model.predict(self.X_train_processed)
                    errors = np.abs(self.y_train - y_train_pred)
                    
                    # Stimiamo l'errore per un certo livello di confidenza
                    error_percentile = np.percentile(errors, intervallo_confidenza * 100)
                    
                    # Applichiamo questo errore alle nuove predizioni
                    ci_lower = predictions - error_percentile
                    ci_upper = predictions + error_percentile
                    
                    result['ci_lower'] = ci_lower
                    result['ci_upper'] = ci_upper
                else:
                    # Se non abbiamo dati di training, utilizziamo una stima standard
                    rmse = np.sqrt(np.mean((self.y_test - model.predict(self.X_test_processed))**2))
                    z_score = st.norm.ppf(1 - (1 - intervallo_confidenza)/2)
                    
                    ci_lower = predictions - z_score * rmse
                    ci_upper = predictions + z_score * rmse
                    
                    result['ci_lower'] = ci_lower
                    result['ci_upper'] = ci_upper
            else:
                # Per altri modelli, utilizziamo una stima basata su RMSE
                if hasattr(self, 'y_test') and hasattr(self, 'X_test_processed'):
                    rmse = np.sqrt(mean_squared_error(self.y_test, model.predict(self.X_test_processed)))
                    z_score = st.norm.ppf(1 - (1 - intervallo_confidenza)/2)
                    
                    ci_lower = predictions - z_score * rmse
                    ci_upper = predictions + z_score * rmse
                    
                    result['ci_lower'] = ci_lower
                    result['ci_upper'] = ci_upper
                
        return result

    def predici_da_file(self, file_path, model_name=None, target_column=None, intervallo_confidenza=0.95):
        """
        Carica un nuovo dataset da file ed effettua predizioni con intervallo di confidenza
        
        Args:
            file_path: Percorso del file con i nuovi dati
            model_name: Nome del modello da utilizzare
            target_column: Nome della colonna target nel nuovo dataset (se presente)
            intervallo_confidenza: Livello di confidenza per l'intervallo (default: 0.95 per 95%)
            
        Returns:
            DataFrame: DataFrame con i dati originali e le predizioni con intervalli di confidenza
        """
        # Carica il nuovo dataset
        if file_path.endswith('.csv'):
            nuovo_df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            nuovo_df = pd.read_excel(file_path)
        else:
            raise ValueError("Formato file non supportato. Utilizzare CSV o Excel.")
        
        print(f"Nuovo dataset caricato: {nuovo_df.shape[0]} righe e {nuovo_df.shape[1]} colonne")
        
        # Separa target se presente
        X_nuovo = nuovo_df
        y_nuovo = None
        
        if target_column and target_column in nuovo_df.columns:
            X_nuovo = nuovo_df.drop(columns=[target_column])
            y_nuovo = nuovo_df[target_column]
            print(f"Colonna target '{target_column}' trovata e separata")
        
        X_nuovo = self.preprocess_data_prediction(X_nuovo)
        
        # Effettua le predizioni con intervallo di confidenza
        predictions = self.predici(X_nuovo, model_name, intervallo_confidenza=intervallo_confidenza)
        
        # Crea un nuovo DataFrame con i dati originali e le predizioni
        result_df = nuovo_df.copy()
        
        # Aggiungi la predizione principale
        result_df['predizione'] = predictions['predictions']
        
        # Aggiungi l'intervallo di confidenza per la regressione
        if self.problem_type == 'regressione' and 'ci_lower' in predictions and 'ci_upper' in predictions:
            result_df['predizione_min'] = predictions['ci_lower']
            result_df['predizione_max'] = predictions['ci_upper']
            
            # Calcoliamo anche l'ampiezza dell'intervallo di confidenza
            result_df['ic_ampiezza'] = result_df['predizione_max'] - result_df['predizione_min']
            
            print(f"Intervallo di confidenza al {intervallo_confidenza*100}% aggiunto alle predizioni")
        
        # Aggiungi colonne per le probabilità di ogni classe per classificazione
        if self.problem_type == 'classificazione' and 'probabilities' in predictions:
            for i, col in enumerate(sorted(np.unique(self.y_train))):
                result_df[f'prob_classe_{col}'] = predictions['probabilities'][:, i]
                
            # Aggiungiamo una stima dell'incertezza basata sull'entropia della distribuzione di probabilità
            if 'probabilities' in predictions:
                from scipy.stats import entropy
                # Calcola l'entropia delle probabilità (maggiore entropia = maggiore incertezza)
                result_df['incertezza'] = [entropy(prob) for prob in predictions['probabilities']]
        
        # Se abbiamo il target reale, calcoliamo le metriche
        if y_nuovo is not None:
            if self.problem_type == 'classificazione':
                accuracy = accuracy_score(y_nuovo, result_df['predizione'])
                print(f"Accuracy sul nuovo dataset: {accuracy:.4f}")
                
                # Report di classificazione
                print("\nReport di classificazione:")
                print(classification_report(y_nuovo, result_df['predizione']))
                
                # Confusion matrix
                # plt.figure(figsize=(8, 6))
                # cm = confusion_matrix(y_nuovo, result_df['predizione'])
                # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                # plt.title('Matrice di Confusione')
                # plt.xlabel('Predizioni')
                # plt.ylabel('Valori Reali')
                # plt.show()
            else:
                rmse = np.sqrt(mean_squared_error(y_nuovo, result_df['predizione']))
                r2 = r2_score(y_nuovo, result_df['predizione'])
                print(f"RMSE sul nuovo dataset: {rmse:.4f}")
                print(f"R² score: {r2:.4f}")
                
                # Calcoliamo quante volte il valore reale è all'interno dell'intervallo di confidenza
                if 'predizione_min' in result_df.columns and 'predizione_max' in result_df.columns:
                    dentro_ic = ((y_nuovo >= result_df['predizione_min']) & 
                                (y_nuovo <= result_df['predizione_max'])).mean()
                    print(f"Percentuale di valori reali all'interno dell'intervallo di confidenza: {dentro_ic*100:.2f}%")
                    
                    # Aggiungiamo una colonna che indica se il valore reale è dentro l'intervallo
                    result_df['dentro_ic'] = ((y_nuovo >= result_df['predizione_min']) & 
                                            (y_nuovo <= result_df['predizione_max']))
                
                # Visualizza predizioni vs valori reali con intervallo di confidenza
                # plt.figure(figsize=(10, 6))
                # plt.scatter(y_nuovo, result_df['predizione'], alpha=0.5, label='Predizioni')
                
                # Aggiungiamo le barre di errore per l'intervallo di confidenza
                if 'predizione_min' in result_df.columns and 'predizione_max' in result_df.columns:
                    # Ordiniamo i punti per valore reale per una visualizzazione più pulita
                    y_sorted = np.sort(y_nuovo)
                    indices = np.argsort(y_nuovo)
                    
                    pred_sorted = result_df['predizione'].values[indices]
                    lower_sorted = result_df['predizione_min'].values[indices]
                    upper_sorted = result_df['predizione_max'].values[indices]
                    
                    # Plot degli intervalli di confidenza (mostriamo solo un sottoinsieme per chiarezza)
                    n_points = len(y_sorted)
                    step = max(1, n_points // 100)  # Mostra al massimo 100 intervalli
                    
                    # for i in range(0, n_points, step):
                    #     plt.plot([y_sorted[i], y_sorted[i]], 
                    #             [lower_sorted[i], upper_sorted[i]], 
                    #             'r-', alpha=0.3)
                
                # plt.plot([min(y_nuovo), max(y_nuovo)], [min(y_nuovo), max(y_nuovo)], 'k--', label='Ideale')
                # plt.xlabel('Valori Reali')
                # plt.ylabel('Predizioni')
                # plt.title(f'Predizioni vs Valori Reali con Intervallo di Confidenza {intervallo_confidenza*100}%')
                # plt.legend()
                # plt.show()
                
                # Visualizza anche una distribuzione dell'ampiezza degli intervalli di confidenza
                # if 'ic_ampiezza' in result_df.columns:
                #     plt.figure(figsize=(10, 6))
                #     sns.histplot(result_df['ic_ampiezza'], kde=True)
                #     plt.title(f'Distribuzione dell\'Ampiezza degli Intervalli di Confidenza {intervallo_confidenza*100}%')
                #     plt.xlabel('Ampiezza dell\'Intervallo')
                #     plt.ylabel('Frequenza')
                #     plt.show()
        
        return result_df

    def esporta_predizioni(self, predictions_df, output_path=None, format='csv'):
        """
        Esporta le predizioni in un file CSV o Excel
        
        Args:
            predictions_df: DataFrame con le predizioni
            output_path: Percorso dove salvare il file
            format: Formato del file ('csv' o 'excel')
            
        Returns:
            str: Percorso del file salvato
        """
        if output_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_{timestamp}.{format}"
        
        if format == 'csv':
            predictions_df.to_csv(output_path, index=False)
        elif format == 'excel':
            predictions_df.to_excel(output_path, index=False)
        else:
            raise ValueError("Formato non supportato. Utilizzare 'csv' o 'excel'")
        
        print(f"Predizioni esportate come {output_path}")
        return output_path

    
        


    def analisi_sensibilita_cluster(self, cluster_df, parametri_variabili, target, modello=None, 
                           n_punti=20, normalizza=True, plot_3d=False):
        """
        Analizza come la variazione di parametri selezionati influenza il valore target all'interno di un cluster.
        
        Args:
            cluster_df: DataFrame contenente i dati del cluster da analizzare
            parametri_variabili: Lista di parametri da variare nell'analisi di sensibilità
            target: Nome della colonna target da predire
            modello: Modello pre-addestrato da utilizzare (se None, usa self.best_model)
            n_punti: Numero di punti da utilizzare nell'intervallo di variazione per ogni parametro
            normalizza: Se normalizzare i parametri di variazione (min-max)
            plot_3d: Se generare plot 3D quando si variano 2 parametri
            
        Returns:
            DataFrame con i risultati dell'analisi di sensibilità
        """
        import itertools
        from tqdm.notebook import tqdm
        
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("Nessun modello disponibile. Addestrare prima un modello.")
        
        if len(parametri_variabili) > 3:
            print("Attenzione: sono stati specificati più di 3 parametri. "
                "L'analisi potrebbe richiedere molto tempo e i risultati potrebbero essere difficili da visualizzare.")
        
        # Creare un "punto base" usando i valori medi del cluster per tutte le features
        X_base = cluster_df.drop(columns=[target] if target in cluster_df.columns else [])
        # X_processed = pd.DataFrame(self.preprocess_data_prediction(X_base))
        # X_processed.columns = X_base.columns

        punto_base = X_base.mean().to_dict()
        print(f"Punto base (valori medi del cluster):")
        for k, v in punto_base.items():
            if k in parametri_variabili:
                print(f"  {k}: {v:.4f}")
        
        # Determina i range per i parametri da variare
        ranges = {}
        for param in parametri_variabili:
            if param not in X_base.columns:
                raise ValueError(f"Il parametro '{param}' non è presente nel DataFrame")
            
            min_val = X_base[param].min()
            max_val = X_base[param].max()
            
            # Espandi leggermente il range per vedere l'effetto potenziale oltre i limiti del cluster
            range_size = max_val - min_val
            # min_val = max(min_val - range_size * 0.1, 0 if X_base[param].min() >= 0 else min_val * 1.1)
            # max_val = max_val + range_size * 0.1
            
            ranges[param] = np.linspace(min_val, max_val, n_punti)
            print(f"Range per {param}: {min_val:.4f} - {max_val:.4f}")
        
        # Preparare la struttura dati per i risultati
        if len(parametri_variabili) == 1:
            # Caso monodimensionale
            param = parametri_variabili[0]
            results = []
            
            for val in tqdm(ranges[param], desc=f"Variazione di {param}"):
                # Creare un punto di test basato sul punto base
                punto_test = punto_base.copy()
                punto_test[param] = val
                
                # Crea DataFrame per il punto
                df_test = pd.DataFrame([punto_test])
                
                # Preprocessing e predizione
                # X_processed = self.preprocess_data_prediction(df_test)
                X_processed = df_test
                pred_result = self.predici(X_processed, modello)
                
                prediction = pred_result['predictions'][0]
                ci_lower = pred_result.get('ci_lower', [None])[0]
                ci_upper = pred_result.get('ci_upper', [None])[0]
                
                results.append({
                    param: val,
                    'predizione': prediction,
                    'predizione_min': ci_lower,
                    'predizione_max': ci_upper
                })
            
            results_df = pd.DataFrame(results)
            
            # Visualizzazione
            # plt.figure(figsize=(10, 6))
            # plt.plot(results_df[param], results_df['predizione'], 'b-', label='Predizione')
            
            # if 'predizione_min' in results_df.columns and results_df['predizione_min'].notna().all():
            #     plt.fill_between(
            #         results_df[param], 
            #         results_df['predizione_min'], 
            #         results_df['predizione_max'], 
            #         alpha=0.2, 
            #         color='blue', 
            #         label='Intervallo di Confidenza'
            #     )
            
            # plt.xlabel(param)
            # plt.ylabel(target)
            # plt.title(f"Effetto della variazione di {param} su {target}")
            # plt.grid(True, alpha=0.3)
            # plt.legend()
            # plt.show()
            
            # Calcola l'elasticità (sensitivity)
            param_range = results_df[param].max() - results_df[param].min()
            pred_range = results_df['predizione'].max() - results_df['predizione'].min()
            
            if normalizza and param_range > 0 and pred_range > 0:
                # Elasticità normalizzata (variazione % dell'output / variazione % dell'input)
                param_mid = results_df[param].median()
                pred_mid = results_df['predizione'].median()
                
                elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                print(f"Elasticità normalizzata di {target} rispetto a {param}: {elasticity:.4f}")
                print(f"Interpretazione: Una variazione dell'1% in {param} causa una variazione del {elasticity:.4f}% in {target}")
        
        elif len(parametri_variabili) == 2:
            # Caso bidimensionale
            param1, param2 = parametri_variabili
            results = []
            
            for val1, val2 in tqdm(list(itertools.product(ranges[param1], ranges[param2])), 
                                desc=f"Variazione di {param1} e {param2}"):
                # Creare un punto di test
                punto_test = punto_base.copy()
                # punto_test = X_processed.mean().to_dict()
                punto_test[param1] = val1
                punto_test[param2] = val2
                
                # Crea DataFrame per il punto
                df_test = pd.DataFrame([punto_test])
                
                # Preprocessing e predizione
                # X_processed = self.preprocess_data_prediction(df_test)
                X_processed = df_test
                pred_result = self.predici(X_processed, modello)
                
                prediction = pred_result['predictions'][0]
                ci_lower = pred_result.get('ci_lower', [None])[0]
                ci_upper = pred_result.get('ci_upper', [None])[0]
                
                results.append({
                    param1: val1,
                    param2: val2,
                    'predizione': prediction,
                    'predizione_min': ci_lower,
                    'predizione_max': ci_upper
                })
            
            results_df = pd.DataFrame(results)
            
            # # Visualizzazione mappa di calore
            pivot_table = results_df.pivot_table(
                index=param1, 
                columns=param2, 
                values='predizione'
            )
            
            # plt.figure(figsize=(12, 8))
            # sns.heatmap(pivot_table, cmap='viridis', annot=False, cbar_kws={'label': target})
            # plt.title(f"Effetto della variazione di {param1} e {param2} su {target}")
            # plt.tight_layout()
            # plt.show()
            
            # Se richiesto, genera un grafico 3D
            # if plot_3d:
            #     from mpl_toolkits.mplot3d import Axes3D
                
            #     fig = plt.figure(figsize=(12, 10))
            #     ax = fig.add_subplot(111, projection='3d')
                
            #     x = results_df[param1].values
            #     y = results_df[param2].values
            #     z = results_df['predizione'].values
                
            #     # Crea una superficie trisurf
            #     surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.7)
                
            #     ax.set_xlabel(param1)
            #     ax.set_ylabel(param2)
            #     ax.set_zlabel(target)
            #     ax.set_title(f"Superficie di risposta 3D: Effetto di {param1} e {param2} su {target}")
                
            #     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=target)
            #     plt.tight_layout()
            #     plt.show()
                
            # Calcolo delle derivate parziali (gradient)
            print("\nAnalisi di sensibilità:")
            
            for param in [param1, param2]:
                other_param = param2 if param == param1 else param1
                other_param_med = results_df[other_param].median()
                
                # Filtra i risultati con l'altro parametro vicino alla mediana
                filtered = results_df[np.isclose(results_df[other_param], other_param_med, rtol=0.1)]
                
                if len(filtered) >= 2:
                    param_range = filtered[param].max() - filtered[param].min()
                    pred_range = filtered['predizione'].max() - filtered['predizione'].min()
                    
                    if normalizza and param_range > 0 and pred_range > 0:
                        # Elasticità normalizzata
                        param_mid = filtered[param].median()
                        pred_mid = filtered['predizione'].median()
                        
                        elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                        print(f"Elasticità di {target} rispetto a {param} (con {other_param} ≈ {other_param_med:.4f}): {elasticity:.4f}")
        
        else:
            # Caso multidimensionale - qui facciamo un'analisi one-at-a-time
            results = []
            
            for param in parametri_variabili:
                print(f"\nAnalisi di sensibilità per {param}:")
                
                for val in tqdm(ranges[param], desc=f"Variazione di {param}"):
                    # Creare un punto di test
                    # punto_test = punto_base.copy()
                    punto_test = X_processed.mean().to_dict()
                    punto_test[param] = val
                    
                    # Crea DataFrame per il punto
                    df_test = pd.DataFrame([punto_test])
                    
                    # Preprocessing e predizione
                    X_processed = self.preprocess_data_prediction(df_test)
                    pred_result = self.predici(X_processed, modello)
                    
                    prediction = pred_result['predictions'][0]
                    ci_lower = pred_result.get('ci_lower', [None])[0]
                    ci_upper = pred_result.get('ci_upper', [None])[0]
                    
                    result_record = {'parametro_variato': param}
                    result_record.update({p: punto_base[p] for p in parametri_variabili})
                    result_record[param] = val
                    result_record['predizione'] = prediction
                    result_record['predizione_min'] = ci_lower
                    result_record['predizione_max'] = ci_upper
                    
                    results.append(result_record)
                
                # Visualizza il grafico per questo parametro
                param_results = pd.DataFrame([r for r in results if r['parametro_variato'] == param])
                
                # plt.figure(figsize=(10, 6))
                # plt.plot(param_results[param], param_results['predizione'], 'b-', label='Predizione')
                
                # if 'predizione_min' in param_results.columns and param_results['predizione_min'].notna().all():
                #     plt.fill_between(
                #         param_results[param], 
                #         param_results['predizione_min'], 
                #         param_results['predizione_max'], 
                #         alpha=0.2, 
                #         color='blue', 
                #         label='Intervallo di Confidenza'
                #     )
                    
                # plt.xlabel(param)
                # plt.ylabel(target)
                # plt.title(f"Effetto della variazione di {param} su {target}")
                # plt.grid(True, alpha=0.3)
                # plt.legend()
                # plt.show()
                
                # Calcola l'elasticità per questo parametro
                param_range = param_results[param].max() - param_results[param].min()
                pred_range = param_results['predizione'].max() - param_results['predizione'].min()
                
                if normalizza and param_range > 0 and pred_range > 0:
                    param_mid = param_results[param].median()
                    pred_mid = param_results['predizione'].median()
                    
                    elasticity = (pred_range / pred_mid) / (param_range / param_mid)
                    print(f"Elasticità di {target} rispetto a {param}: {elasticity:.4f}")
            
            # Crea una tabella riassuntiva di sensibilità
            sensitivity_summary = []
            
            for param in parametri_variabili:
                param_results = pd.DataFrame([r for r in results if r['parametro_variato'] == param])
                param_range = param_results[param].max() - param_results[param].min()
                pred_range = param_results['predizione'].max() - param_results['predizione'].min()
                
                if normalizza and param_range > 0 and pred_range > 0:
                    param_mid = param_results[param].median()
                    pred_mid = param_results['predizione'].median()
                    elasticity = abs((pred_range / pred_mid) / (param_range / param_mid))
                else:
                    elasticity = abs(pred_range / param_range) if param_range > 0 else 0
                    
                sensitivity_summary.append({
                    'parametro': param,
                    'elasticità': elasticity,
                    'variazione_assoluta': pred_range
                })
            
            sensitivity_df = pd.DataFrame(sensitivity_summary).sort_values('elasticità', ascending=False)
            print("\nRiepilogo della sensibilità (ordinato per importanza):")
            print(sensitivity_df)
            
            # Grafico a barre dell'elasticità
            # plt.figure(figsize=(10, 6))
            # sns.barplot(x='parametro', y='elasticità', data=sensitivity_df)
            # plt.title(f"Elasticità di {target} rispetto ai parametri")
            # plt.xticks(rotation=45)
            # plt.tight_layout()
            # plt.show()
        
        return results_df
    
    def ottimizza_parametri_cluster(self, cluster_df, parametri_variabili, target, objective='min',
                             modello=None, vincoli=None, n_trials=100, verbose=True):
        """
        Ottimizza i parametri all'interno di un cluster per massimizzare o minimizzare un target.
        
        Args:
            cluster_df: DataFrame contenente i dati del cluster da analizzare
            parametri_variabili: Lista di parametri da ottimizzare
            target: Nome della colonna target da ottimizzare
            objective: 'min' per minimizzare il target, 'max' per massimizzarlo
            modello: Modello pre-addestrato da utilizzare (se None, usa self.best_model)
            vincoli: Dizionario con i vincoli per ciascun parametro {'param': (min, max)}
            n_trials: Numero di iterazioni per l'ottimizzazione
            verbose: Se mostrare i progressi dell'ottimizzazione
            
        Returns:
            dict: Parametri ottimizzati e valore target predetto
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("Per utilizzare questa funzione, installa optuna: pip install optuna")
            return None
        
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("Nessun modello disponibile. Addestrare prima un modello.")
        
        # Creare un punto base usando i valori medi del cluster per tutte le features
        X_base = cluster_df.drop(columns=[target] if target in cluster_df.columns else [])
        punto_base = X_base.mean().to_dict()

        X_processed = pd.DataFrame(self.preprocess_data_prediction(X_base))
        X_processed.columns = X_base.columns
        
        # Determina i range per i parametri da ottimizzare
        if vincoli is None:
            vincoli = {}
            for param in parametri_variabili:
                if param not in X_base.columns:
                    raise ValueError(f"Il parametro '{param}' non è presente nel DataFrame")
                
                min_val = X_base[param].min()
                max_val = X_base[param].max()
                
                # Espandi leggermente il range
                range_size = max_val - min_val
                min_val = max(min_val - range_size * 0.1, 0 if X_base[param].min() >= 0 else min_val * 1.1)
                max_val = max_val + range_size * 0.1
                
                vincoli[param] = (min_val, max_val)
        
        # Definisci la funzione obiettivo per Optuna
        def objective_func(trial):
            # Creare un punto di test basato sul punto base
            punto_test = punto_base.copy()
            # punto_test = X_processed.mean().to_dict()
            
            # Suggerisci valori per ciascun parametro
            for param in parametri_variabili:
                min_val, max_val = vincoli[param]
                punto_test[param] = trial.suggest_float(param, min_val, max_val)
            
            # Crea DataFrame per il punto
            df_test = pd.DataFrame([punto_test])
            
            # Preprocessing e predizione
            # X_processed = self.preprocess_data_prediction(df_test)
            X_processed = df_test
            pred_result = self.predici(X_processed, modello)
            
            prediction = pred_result['predictions'][0]
            
            # Se minimizziamo, restituisci direttamente la predizione
            # Se massimizziamo, restituisci il negativo della predizione (Optuna minimizza)
            return prediction if objective == 'min' else -prediction
        
        # Configura e avvia l'ottimizzazione
        sampler = TPESampler(seed=42)  # Per riproducibilità
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=verbose)
        
        # Ottieni i migliori parametri
        best_params = study.best_params
        
        # Crea un punto con i migliori parametri
        punto_ottimizzato = punto_base.copy()
        punto_ottimizzato.update(best_params)
        
        # Calcola il valore target predetto
        df_ottimizzato = pd.DataFrame([punto_ottimizzato])
        X_processed = self.preprocess_data_prediction(df_ottimizzato)
        X_processed = df_ottimizzato
        pred_result = self.predici(X_processed, modello)
        predicted_target = pred_result['predictions'][0]
        
        # Visualizza risultati
        if verbose:
            print(f"\nOttimizzazione completata per {objective}imizzare {target}")
            print(f"Valore {target} predetto: {predicted_target:.4f}")
            print("\nParametri ottimizzati:")
            for param, value in best_params.items():
                orig_value = punto_base[param]
                change_pct = ((value - orig_value) / orig_value) * 100 if orig_value != 0 else float('inf')
                print(f"  {param}: {value:.4f} (originale: {orig_value:.4f}, variazione: {change_pct:.2f}%)")
            
            # Plot dell'andamento dell'ottimizzazione
            # plt.figure(figsize=(10, 6))
            # optuna.visualization.matplotlib.plot_optimization_history(study)
            # plt.title(f"Andamento dell'ottimizzazione per {objective}imizzare {target}")
            # plt.tight_layout()
            # plt.show()
            
            # # Plot dell'importanza dei parametri
            # plt.figure(figsize=(10, 6))
            # try:
            #     optuna.visualization.matplotlib.plot_param_importances(study)
            #     plt.title(f"Importanza dei parametri per {objective}imizzare {target}")
            #     plt.tight_layout()
            #     plt.show()
            # except:
            #     print("Impossibile calcolare l'importanza dei parametri")
        
        result = {
            'parametri_ottimizzati': best_params,
            'target_predetto': predicted_target,
            'target_originale': punto_base.get(target, None),
            'punto_completo': punto_ottimizzato,
            'study': study  # Restituisce l'oggetto study per analisi future
        }
        
        return result

    def confronta_scenari_cluster(self, cluster_df, scenari, target, modello=None):
        """
        Confronta diversi scenari di parametri e il loro effetto sul target.
        
        Args:
            cluster_df: DataFrame contenente i dati del cluster da analizzare
            scenari: Lista di dizionari, ciascuno rappresenta uno scenario {'nome': 'Scenario 1', 'parametri': {'param1': val1, ...}}
            target: Nome della colonna target da predire
            modello: Modello pre-addestrato da utilizzare (se None, usa self.best_model)
            
        Returns:
            DataFrame: Confronto degli scenari con i valori target predetti
        """
        if modello is None and self.best_model is not None:
            modello = self.models[self.best_model]
        elif modello is None:
            raise ValueError("Nessun modello disponibile. Addestrare prima un modello.")
        
        # Creare un punto base usando i valori medi del cluster per tutte le features
        X_base = cluster_df.drop(columns=[target] if target in cluster_df.columns else [])
        punto_base = X_base.mean().to_dict()
        X_processed = pd.DataFrame(self.preprocess_data_prediction(X_base))
        X_processed.columns = X_base.columns

        # Prepara il punto base come scenario
        scenari_con_base = [{'nome': 'Base (media cluster)', 'parametri': {}}] + scenari
        
        # Lista per i risultati
        risultati = []
        
        # Valuta ciascuno scenario
        for scenario in scenari_con_base:
            # Crea un punto di test basato sul punto base
            punto_test = punto_base.copy()
            # punto_test = X_processed.mean().to_dict()
            
            # Aggiorna con i parametri dello scenario (per lo scenario base non ci sono modifiche)
            punto_test.update(scenario['parametri'])
            
            # Crea DataFrame per il punto
            df_test = pd.DataFrame([punto_test])
            
            # Preprocessing e predizione
            # X_processed = self.preprocess_data_prediction(df_test)
            X_processed = df_test
            pred_result = self.predici(X_processed, modello)
            
            prediction = pred_result['predictions'][0]
            ci_lower = pred_result.get('ci_lower', [None])[0]
            ci_upper = pred_result.get('ci_upper', [None])[0]
            
            # Risultato per questo scenario
            result = {
                'scenario': scenario['nome'],
                'predizione': prediction,
                'predizione_min': ci_lower,
                'predizione_max': ci_upper
            }
            
            # Aggiungi i valori dei parametri
            for param, value in punto_test.items():
                if param in set().union(*[s['parametri'].keys() for s in scenari if 'parametri' in s]):
                    result[f"param_{param}"] = value
            
            risultati.append(result)
        
        results_df = pd.DataFrame(risultati)
        
        # Calcola le variazioni percentuali rispetto allo scenario base
        base_prediction = results_df.loc[results_df['scenario'] == 'Base (media cluster)', 'predizione'].values[0]
        results_df['variazione_pct'] = ((results_df['predizione'] - base_prediction) / base_prediction) * 100
        
        # Visualizzazione dei risultati
        # plt.figure(figsize=(12, 6))
        # ax = sns.barplot(x='scenario', y='predizione', data=results_df)
        
        # Aggiungi le barre di errore se disponibili
        if 'predizione_min' in results_df.columns and results_df['predizione_min'].notna().all():
            for i, row in results_df.iterrows():
                ax.errorbar(
                    i, row['predizione'], 
                    yerr=[[row['predizione']-row['predizione_min']], [row['predizione_max']-row['predizione']]],
                    fmt='none', capsize=5, color='black', alpha=0.7
                )
        
        # plt.title(f"Confronto degli scenari - Effetto sul {target}")
        # plt.ylabel(target)
        # plt.xticks(rotation=45, ha='right')
        # plt.grid(axis='y', alpha=0.3)
        
        # Aggiungi i valori sopra le barre
        for i, v in enumerate(results_df['predizione']):
            var_pct = results_df['variazione_pct'].iloc[i]
            var_text = f" ({var_pct:+.1f}%)" if i > 0 else ""
            ax.text(i, v + (results_df['predizione'].max() * 0.01), 
                    f"{v:.2f}{var_text}", 
                    ha='center', fontsize=9)
        
        # plt.tight_layout()
        # plt.show()
        
        # Mostra anche una tabella dei risultati
        print("\nRisultati numerici:")
        display_df = results_df[['scenario', 'predizione', 'variazione_pct']].copy()
        display_df['predizione'] = display_df['predizione'].round(4)
        display_df['variazione_pct'] = display_df['variazione_pct'].round(2)
        display_df = display_df.rename(columns={'variazione_pct': 'variazione (%)'})
        print(display_df)
        return results_df

    def esegui_pipeline_completa(self, ottimizza=True, visualizza_feature_importance=True, 
                           salva_modello=True, file_path_salvataggio=None,
                           predici_nuovo_file=True, nuovo_file_path=None, 
                           target_column_nuovo=None, esporta_predizioni=False,
                           formato_export='csv', analisi_sensitività=False, parametri_variabili=None,
                           target="QHnd",
                           ottimizza_parametri_cluster=False, objective='min',confronta_scenari_cluster=False,scenari=None,
                           cluster_df=None,
                           ):
        """
        Esegue l'intera pipeline dall'inizio alla fine, incluso salvataggio modello
        e predizione su nuovi dati
        
        Args:
            ottimizza: Se ottimizzare gli iperparametri
            visualizza_feature_importance: Se visualizzare l'importanza delle feature
            salva_modello: Se salvare il modello migliore
            file_path_salvataggio: Percorso dove salvare il modello (senza estensione)
            predici_nuovo_file: Se effettuare predizioni su un nuovo file di dati
            nuovo_file_path: Percorso del file con i nuovi dati per predizione
            target_column_nuovo: Nome della colonna target nel nuovo dataset (se presente)
            esporta_predizioni: Se esportare le predizioni in un file
            formato_export: Formato del file ('csv' o 'excel')
        
        Returns:
            tuple: (miglior modello, miglior score, file salvati, predizioni)
        """
        self.carica_dati()
        self.analisi_esplorativa()
        self.preprocess_dati()
        self.addestra_modelli()
        
        
        if ottimizza:
            best_model = self.ottimizza_iperparametri()
            
        if visualizza_feature_importance:
            self.feature_importance()
        
        files_salvati = None
        if salva_modello:
            files_salvati = self.salva_modello(
                model=best_model, 
                file_path=file_path_salvataggio
            )
            print(f"\nModello salvato: {files_salvati}")
        
        predizioni_df = None
        if predici_nuovo_file and nuovo_file_path:
            print(f"\nEsecuzione predizioni sul file: {nuovo_file_path}")
            predizioni_df = self.predici_da_file(
                file_path=nuovo_file_path,
                model_name=best_model,
                target_column=target_column_nuovo
            )
            
            if esporta_predizioni and predizioni_df is not None:
                export_path = self.esporta_predizioni(
                    predictions_df=predizioni_df,
                    format=formato_export
                )
                print(f"Predizioni esportate in: {export_path}")
            
        print("\nPipeline completata!")
        print(f"Miglior modello: {self.best_model}")
        if self.problem_type == 'classificazione':
            print(f"Best score (accuracy): {self.best_score:.4f}")
        else:
            print(f"Best score (RMSE): {self.best_score:.4f}")
        
        if analisi_sensitività:
            results_analisi_sensibilita = self.analisi_sensibilita_cluster(
                cluster_df=cluster_df,
                parametri_variabili=parametri_variabili,
                # parametri_variabili=['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance'],
                target=target,
                plot_3d=True,
                modello=best_model
            )

        if ottimizza_parametri_cluster:
            self.ottimizza_parametri_cluster(
                cluster_df=cluster_df,
                parametri_variabili=parametri_variabili,
                target=target,
                objective='min',
                modello=best_model, 
                vincoli=None, n_trials=100, verbose=True
            )

        if confronta_scenari_cluster:
            df_confronto_scenari = self.confronta_scenari_cluster(
            cluster_df=cluster_df, 
            scenari=scenari,
            target=target, 
            modello=best_model
        )

        return self.best_model, self.best_score, files_salvati, predizioni_df, results_analisi_sensibilita, df_confronto_scenari

def run_model(
    file_path_cluster, 
    target_, 
    problem_type_, 
    variables_for_sensitivity_analysis, 
    file_path_save_model,
    confronta_scenari_cluster,
    scenari,
    path_save_result):

    modello = ModelloPredittivo(file_path=file_path_cluster, target_column=target_, problem_type=problem_type_)
    _,_,_,predizioni_df,results_analisi_sensibilita,df_confronto_scenari = modello.esegui_pipeline_completa(
        predici_nuovo_file=False, 
        nuovo_file_path=file_path_cluster, 
        target_column_nuovo=target_,
        file_path_salvataggio = file_path_save_model,
        analisi_sensitività=True,
        parametri_variabili=variables_for_sensitivity_analysis,
        target=target_,
        confronta_scenari_cluster=confronta_scenari_cluster,
        scenari=scenari,
        cluster_df = pd.read_csv(file_path_cluster, sep=',', decimal='.', low_memory=False, header=0),
    )
    # save results
    results_analisi_sensibilita.to_csv(f"{path_save_result}/results_analisi_sensibilita_{cluster_name}.csv", index=False)
    df_confronto_scenari.to_csv(f"{path_save_result}/df_confronto_scenari_{cluster_name}.csv", index=False)

    return predizioni_df,results_analisi_sensibilita,df_confronto_scenari

cluster_name = "cluster_0"
file_path_cluster = f"Dash_app/data_cluster/{cluster_name}.csv"
target_ = "QHnd"
problem_type_ = "regressione"
file_path_save_model = f'Dash_app/models/{cluster_name}'
variables_for_sensitivity_analysis = ['average_opaque_surface_transmittance', 'average_glazed_surface_transmittance']
confronta_scenari_cluster=True
path_save_result = 'Dash_app/result_sensitivity_cluster'
scenari=[
    {'nome': 'Scenario 1', 'parametri': {'average_opaque_surface_transmittance': 0.5, 'average_glazed_surface_transmittance': 1}},
    {'nome': 'Scenario 2', 'parametri': {'average_opaque_surface_transmittance': 0.2, 'average_glazed_surface_transmittance': 0.7}}
]
predizioni_df,results_analisi_sensibilita,df_confronto_scenari = run_model(
    file_path_cluster, 
    target_, 
    problem_type_, 
    variables_for_sensitivity_analysis,
    file_path_save_model,
    confronta_scenari_cluster,
    scenari,
    path_save_result)