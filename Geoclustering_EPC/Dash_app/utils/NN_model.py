import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Impostazione per riproducibilità
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Caricamento e ispezione preliminare del dataset
def load_and_explore_data(file_path):
    """
    Carica e esplora il dataset tabellare.
    
    Args:
        file_path: Percorso del file CSV o Excel
    
    Returns:
        DataFrame con i dati
    """
    # Caricamento dei dati, verificando l'estensione
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, sep=",", decimal=".", low_memory=False, header=0)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Formato file non supportato. Utilizzare CSV o Excel.")
    
    # Esplorazione dei dati
    print(f"Dimensioni del dataset: {df.shape}")
    print("\nPrime 5 righe:")
    print(df.head())
    print("\nInfo sul dataset:")
    print(df.info())
    print("\nStatistiche descrittive:")
    print(df.describe())
    
    # Controllo valori mancanti
    missing_values = df.isnull().sum()
    print("\nValori mancanti per colonna:")
    print(missing_values[missing_values > 0])
    
    return df

# Step 2: Preprocessing dei dati
def preprocess_data(df, target_column, unuseful_columns=None,categorical_cols=None, numerical_cols=None, problem_type='regression'):
    """
    Preprocessing completo del dataset tabellare.
    
    Args:
        df: DataFrame con i dati
        target_column: Nome della colonna target
        categorical_cols: Lista di colonne categoriche
        numerical_cols: Lista di colonne numeriche
        problem_type: 'regression' o 'classification'
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    """
    # Separazione tra features e target
    X = df.drop(target_column, axis=1)
    X = df.drop(unuseful_columns, axis=1)
    y = df[target_column]
    
    # Identificazione automatica delle colonne categoriche e numeriche se non specificate
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Colonne categoriche: {categorical_cols}")
    print(f"Colonne numeriche: {numerical_cols}")
    
    # Divisione in train, validation e test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    # Creazione di un preprocessore con ColumnTransformer
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Applicazione del preprocessore
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Controllo se è un problema di classificazione e conversione della target se necessario
    if problem_type == 'classification':
        # Verifica se la target è già numerica
        if y_train.dtype == 'object' or y_train.dtype == 'category':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)
            y_test = le.transform(y_test)
            print(f"Classi target codificate: {le.classes_}")
    
    print(f"Dimensioni X_train processato: {X_train_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor

# Step 3: Definizione di una rete neurale avanzata con tecniche moderne
def create_tabular_neural_network(input_dim, output_dim=1, problem_type='regression', use_residual=True):
    """
    Crea una rete neurale avanzata per dati tabellari.
    
    Args:
        input_dim: Dimensionalità dell'input
        output_dim: Dimensionalità dell'output
        problem_type: 'regression' o 'classification'
        use_residual: Usare connessioni residuali
    
    Returns:
        Modello TensorFlow compilato
    """
    # Definizione degli input
    inputs = Input(shape=(input_dim,))
    
    # Layer iniziale
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    
    # Blocco residuale 1
    if use_residual:
        residual = x
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        # Connessione residuale
        x = Concatenate()([x, residual])
    else:
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
    
    # Blocco residuale 2
    if use_residual:
        residual = x
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        # Connessione residuale
        x = Concatenate()([x, residual])
    else:
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
    
    # Layer finali
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Layer di output in base al tipo di problema
    if problem_type == 'regression':
        outputs = Dense(output_dim, activation='linear')(x)
        loss = 'mse'
        metrics = ['mae']
    elif problem_type == 'binary_classification':
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:  # multi-class classification
        outputs = Dense(output_dim, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    # Creazione del modello
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compilazione del modello
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    print(model.summary())
    return model

# Step 4: Training con tecniche avanzate
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
    """
    Addestramento del modello con tecniche avanzate.
    
    Args:
        model: Modello TensorFlow compilato
        X_train, y_train: Dati di training
        X_val, y_val: Dati di validazione
        batch_size: Dimensione del batch
        epochs: Massimo numero di epoche
    
    Returns:
        History dell'addestramento e modello addestrato
    """
    # Definizione dei callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Addestramento del modello
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history, model

# Step 5: Valutazione del modello
def evaluate_model(model, X_test, y_test, problem_type='regression'):
    """
    Valutazione del modello addestrato.
    
    Args:
        model: Modello addestrato
        X_test, y_test: Dati di test
        problem_type: 'regression' o 'classification'
    
    Returns:
        Metriche di valutazione
    """
    y_pred = model.predict(X_test)
    
    if problem_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Plot di confronto tra valori reali e predetti
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Valori reali')
        plt.ylabel('Valori predetti')
        plt.title('Confronto tra valori reali e predetti')
        plt.show()
        
        return {'mse': mse, 'rmse': rmse}
    
    elif problem_type == 'binary_classification':
        # Arrotondamento per predizioni binarie
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred_binary)
        report = classification_report(y_test, y_pred_binary)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        
        # Plot della matrice di confusione
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        return {'accuracy': acc, 'report': report, 'confusion_matrix': cm}
    
    else:  # multi-class classification
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        acc = accuracy_score(y_test, y_pred_classes)
        report = classification_report(y_test, y_pred_classes)
        cm = confusion_matrix(y_test, y_pred_classes)
        
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)
        
        # Plot della matrice di confusione
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        return {'accuracy': acc, 'report': report, 'confusion_matrix': cm}

# Step 6: Visualizzazione dell'andamento dell'addestramento
def plot_training_history(history):
    """
    Visualizza l'andamento dell'addestramento.
    
    Args:
        history: History object dal training
    """
    plt.figure(figsize=(12, 4))
    
    # Plot della loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot delle metriche (accuracy per classificazione, MAE per regressione)
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy durante l\'addestramento')
        plt.ylabel('Accuracy')
    elif 'mae' in history.history:
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('MAE durante l\'addestramento')
        plt.ylabel('MAE')
    
    plt.xlabel('Epoche')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Step 7: Funzione principale per l'utilizzo semplificato
def train_neural_network_on_tabular_data(
    file_path,
    target_column,
    categorical_cols=None,
    numerical_cols=None,
    problem_type='regression',
    use_residual=True,
    batch_size=32,
    epochs=100
):
    """
    Funzione principale che combina tutte le operazioni per addestrare una rete neurale su dati tabellari.
    
    Args:
        file_path: Percorso del file CSV o Excel
        target_column: Nome della colonna target
        categorical_cols: Lista di colonne categoriche (opzionale)
        numerical_cols: Lista di colonne numeriche (opzionale)
        problem_type: 'regression', 'binary_classification' o 'multiclass_classification'
        use_residual: Utilizzare connessioni residuali
        batch_size: Dimensione del batch
        epochs: Numero massimo di epoche
    
    Returns:
        Modello addestrato, preprocessore, metriche di valutazione
    """
    # Step 1: Caricamento ed esplorazione
    df = load_and_explore_data(file_path)
    
    # Step 2: Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_data(
        df, target_column, categorical_cols, numerical_cols, problem_type
    )
    
    # Step 3: Creazione del modello
    input_dim = X_train.shape[1]
    
    # Determinazione delle dimensioni di output
    if problem_type == 'regression':
        output_dim = 1
    elif problem_type == 'binary_classification':
        output_dim = 1
    else:  # multiclass_classification
        output_dim = len(np.unique(y_train))
    
    model = create_tabular_neural_network(
        input_dim=input_dim,
        output_dim=output_dim,
        problem_type=problem_type,
        use_residual=use_residual
    )
    
    # Step 4: Training
    history, trained_model = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=batch_size, epochs=epochs
    )
    
    # Step 5: Valutazione
    metrics = evaluate_model(trained_model, X_test, y_test, problem_type)
    
    # Step 6: Visualizzazione dell'andamento
    plot_training_history(history)
    
    return trained_model, preprocessor, metrics


# Esempio di utilizzo
if __name__ == "__main__":

    # Sostituire con il percorso del tuo dataset
    file_path = "clustering.csv"
    
    # Nome della colonna target
    target_column = "EPh"

    # Column to delete
    unuseful_columns = ["Unnamed: 0", "EPh", "EPgl","EPl", "EPt", "EPc", "EPv", "EPw"]    
    
    # Tipo di problema ('regression', 'binary_classification', 'multiclass_classification')
    problem_type = "regression"
    
    # Addestramento del modello
    model, preprocessor, metrics = train_neural_network_on_tabular_data(
        file_path=file_path,
        target_column=target_column,
        problem_type=problem_type,
        epochs=50
    )
    
    # Salvataggio del modello
    model.save("tabular_neural_network_model")
    
    # Per il preprocessore, possiamo utilizzare joblib
    import joblib
    joblib.dump(preprocessor, "tabular_preprocessor.joblib")
    
    print("Modello addestrato e salvato con successo!")