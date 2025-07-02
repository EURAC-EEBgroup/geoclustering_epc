import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import dash_mantine_components as dmc
# k-MEANS Algorithm 
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import shutil
import time

def elbow_silhouette_method(df_, columns_selected, cluster_method_custom, cluster_value:int=None, cluster_method_stat:str="elbow"):
    # Subset df according to the columns selected by the user
    df = df_.loc[:, columns_selected].dropna()

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Silhouette and elbow score
    # Use Silhouette and Elbow Analysis to find the optimal number of clusters
    silhouette_scores = []
    inertia = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0, n_init='auto')
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        inertia.append(kmeans.inertia_)

    # Transform the two lists into a list of dictionaries with rounded inertia values
    data_elbow = []
    for k_value, inertia_value in zip(K, inertia):
        data_elbow.append({
            "K": k_value,
            "inertia": round(inertia_value, 2)
        })

    data_silhouette = []
    for k_value, silhouette_value in zip(K, silhouette_scores):
        data_silhouette.append({
            "K": k_value,
            "silhouette": round(silhouette_value, 2)
        })

    # Find the elbow point (simplified method)
    changes = np.diff(inertia)
    second_derivative = np.diff(changes)
    elbow_index = np.argmax(np.abs(second_derivative)) + 2  # +2 due to double differencing
    optimal_k_elbow = K[elbow_index]

    # Find the max silhouette score
    optimal_k_silhouette = K[np.argmax([round(float(val), 2) for val in silhouette_scores])]+ 2 

    linechart_elbow = dmc.LineChart(
        h=300,
        data=data_elbow,
        series=[{"name": "inertia", "label": "Inertia"}],
        dataKey="K",
        type="gradient",
        strokeWidth=5,
        curveType="natural",
        p="lg",
        referenceLines=[
            {"x": optimal_k_elbow, "label": "Best Cluster"},
        ],
    )

    linechart_silhouette = dmc.LineChart(
        h=300,
        data=data_silhouette,
        series=[{"name": "silhouette", "label": "Silhouette Score"}],
        dataKey="K",
        type="gradient",
        strokeWidth=5,
        curveType="natural",
        p="lg",
        referenceLines=[
            {"x": optimal_k_silhouette, "label": "Best Cluster"},
        ],
    )

    fieldset_elbow = dmc.Fieldset(
        children = [
            linechart_elbow,
        ],
        legend="Elbow Method"
    )

    fieldset_silhouette = dmc.Fieldset(
        children = [
            linechart_silhouette,
        ],
        legend="Silhouette Method"
    )

    if cluster_method_custom == True:
        optimal_k = cluster_value
    else: 
        if cluster_method_stat == "elbow":
            optimal_k = optimal_k_elbow
        else:
            optimal_k = optimal_k_silhouette
    # Cluster data with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0, n_init='auto')
    kmeans.fit(X_scaled)

    # Plot the clusters and their centers
    centers = kmeans.cluster_centers_
    labels_cluster = kmeans.labels_
    df_['cluster'] = labels_cluster.astype('str')
    
    return fieldset_elbow, fieldset_silhouette, optimal_k_silhouette, X_scaled, df_, optimal_k


def pulisci_e_ricrea_file(cartella):
    """
    Controlla se nella cartella specificata ci sono file.
    Se sì, li cancella tutti e poi crea i nuovi file specificati.
    
    Args:
        cartella (str): Percorso della cartella da controllare/pulire
        nuovi_file (list, optional): Lista di nomi dei nuovi file da creare
            Default None crea file di esempio
    """
    # Verifica che la cartella esista
    if not os.path.exists(cartella):
        print(f"La cartella {cartella} non esiste. La sto creando...")
        os.makedirs(cartella)
        print(f"Cartella {cartella} creata con successo.")
    
    # Controlla se ci sono file nella cartella
    contenuto = os.listdir(cartella)
    file_presenti = [f for f in contenuto if os.path.isfile(os.path.join(cartella, f))]
    
    # Se ci sono file, cancellali
    if file_presenti:
        print(f"Trovati {len(file_presenti)} file nella cartella {cartella}:")
        for file in file_presenti:
            file_path = os.path.join(cartella, file)
            try:
                os.remove(file_path)
                print(f"  - Cancellato: {file}")
            except Exception as e:
                print(f"  - Errore nella cancellazione di {file}: {str(e)}")
    else:
        print(f"Nessun file trovato nella cartella {cartella}")

def generate_dataset_cluster(
    df_, 
    columns_selected, 
    cluster_method_custom:bool=False, 
    cluster_value:int=2, 
    cluster_method_stat:str="elbow",
    delete_columns:bool=False,
    column_to_delete = [],
    save_df_cluster:bool=False,
    path_folder_save_df_cluster:str=""):
    
    # Rimuovi data dove average_opaque_surface_transmittance è 0.1
    df=df_.copy()
    df = df[df['average_opaque_surface_transmittance'] > 0.1]
    # Rimuovi data dove average_glazed_surface_transmittance è minore di 0.5
    df = df[df['average_glazed_surface_transmittance'] > 0.5]
    
    elbow_method_graph,silhouette_method_graph,_,_,df_cluster, optimal_k = elbow_silhouette_method(df, columns_selected, cluster_method_custom, cluster_value, cluster_method_stat)
    if save_df_cluster:
        pulisci_e_ricrea_file(path_folder_save_df_cluster)
        for cluster in range(optimal_k):
            df_cluster_ = df_cluster[df_cluster['cluster'] == str(cluster)]
            if delete_columns:
                df_cluster_ = df_cluster_.select_dtypes(include=[np.number]).drop(columns=column_to_delete)
            df_cluster_.to_csv(f"{path_folder_save_df_cluster}/cluster_{cluster}.csv", sep=",", decimal=".",index=False)
    
    return df_cluster, optimal_k


