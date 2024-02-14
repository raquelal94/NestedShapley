import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import numpy as np

def ClustersTuples(row):
    row_list = list(row)
    row_list = [int(float(x)) for x in row_list if x != 100]
    row_tuple = (0,) + tuple(row_list)
    return row_tuple

# Functions
def GenerateArray(data):

    data = data.resample("H").sum()
    data = data.stack()
    data.index.names = ["time", "house"]
    # Total Consumption
    dataTotalConsumption = data.groupby(level=1).sum() #in kWh
    # Average Monthly  peak
    dataAveragePeakPerMonth = data.groupby([data.index.get_level_values(0).month, "house"]).max()
    dataAveragePeakPerMonth = dataAveragePeakPerMonth.groupby("house").mean()
    # Concat both dataframes
    df = pd.concat([dataTotalConsumption,
                    dataAveragePeakPerMonth], axis = 1)
    df.columns = ["Total_Consumption", "Ave_Monthly_Peak"]

    return df.to_numpy(), df

def SilhouetteScoreRange(array, nClusters):

    preprocessor = Pipeline([("scaler", StandardScaler())])

    clusterer = Pipeline(
        [
            ("kmeans",
             KMeans(
                 n_clusters=nClusters,
                 init="k-means++",
                 n_init=50,
                 max_iter=500,
                 random_state=None
             ))
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )

    pipe.fit(array)

    preprocessed_data = pipe["preprocessor"].transform(array)
    predicted_labels = pipe["clusterer"]["kmeans"].labels_

    sil_score = silhouette_score(preprocessed_data, predicted_labels)
    sample_silhouette_values = silhouette_samples(preprocessed_data, predicted_labels)
    centers = pipe["clusterer"]["kmeans"].cluster_centers_

    return sil_score, sample_silhouette_values, centers

def KMeansAlgorithm(array, dataframe, nClusters):
    component_1 = "Total_Consumption"
    component_2 = "Ave_Monthly_Peak"

    preprocessor = Pipeline([("scaler", StandardScaler())])

    clusterer = Pipeline(
        [
            ("kmeans",
             KMeans(
                 n_clusters=nClusters,
                 init="k-means++",
                 n_init=50,
                 max_iter=500,
                 random_state=None
             ))
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )

    pipe.fit(array)

    preprocessed_data = pipe["preprocessor"].transform(array)
    predicted_labels = pipe["clusterer"]["kmeans"].labels_

    # sil_score = silhouette_score(preprocessed_data, predicted_labels)
    # sample_silhouette_values = silhouette_samples(preprocessed_data, predicted_labels)
    # centers = pipe["clusterer"]["kmeans"].cluster_centers_

    dataKMeans = pd.DataFrame(
        array,
        columns=[component_1, component_2])
    dataKMeans["predicted cluster"] = predicted_labels
    dataKMeans.index = dataframe.index

    dataProcessed = pd.DataFrame(
        preprocessed_data,
        columns=[component_1, component_2])
    dataProcessed["predicted cluster"] = predicted_labels
    dataProcessed.index = dataframe.index

    return dataKMeans, dataProcessed
def CheckNumberHouseholds(filtered_dataframe, node_households, parent_node, cluster):
    number_households = filtered_dataframe.count()[0]
    node_households[parent_node + (cluster,)] = number_households

    if number_households > 10:
        return True
    else:
        return False

def RunCluster(parent_node, cluster, layer, dataframe, filtered_dataframe, range_n_clusters, node_clusters, bool_household, intermediate,
               sil_score_dict, sample_silhouette_values_dict, centers_dict, sample_silhouette_bool_dict, filtered_cluster):
    n_cluster_candidate = 10

    if bool_household:
        array = filtered_dataframe.loc[:,["Total_Consumption", "Ave_Monthly_Peak"]].to_numpy()

        for n_clusters in range_n_clusters:
            sil_score_dict[n_clusters], sample_silhouette_values_dict[n_clusters], centers_dict[n_clusters] = SilhouetteScoreRange(array=array, nClusters=n_clusters)
            bool_list = all(values >= 0 for values in sample_silhouette_values_dict)
            sample_silhouette_bool_dict[n_clusters] = bool_list
            filtered_cluster[n_clusters] = False

            if sil_score_dict[n_clusters] >= 0.5 and bool_list == True:
                filtered_cluster[n_clusters] = True

                if n_cluster_candidate >= n_clusters:  # update the candidate for clustering to always obtain the minimum
                    n_cluster_candidate = n_clusters

        if n_cluster_candidate == 10: #it implies that none value has pass the filter of 0.5 and bool_list == True
            n_cluster_candidate = max(sil_score_dict, key=sil_score_dict.get)

    else:
        n_cluster_candidate = 0

    node_clusters[parent_node + (cluster,)] = n_cluster_candidate  # update the number of clusters in layer

    # Proceed to generate the cluster for layer
    if n_cluster_candidate > 0:
        dataKMeans, dataProcessed = KMeansAlgorithm(array=array, dataframe=filtered_dataframe, nClusters=n_cluster_candidate)
        # Save predicted cluster
        dataframe.loc[dataKMeans.index, f"Layer_{layer}"] = dataKMeans.loc[:, "predicted cluster"]
    else:
        dataframe.loc[filtered_dataframe.index, f"Layer_{layer}"] = None

    list_tuples = [(parent_node + (cluster,) , i) for i in range_n_clusters]
    intermediate_ = pd.DataFrame(index=pd.MultiIndex.from_tuples(list_tuples),
                                 columns=["sil_score", "sample_silhouette_bool", "valid"])

    intermediate_["sil_score"] = [value for key, value in sil_score_dict.items()]
    intermediate_["sample_silhouette_bool"] = [value for key, value in sample_silhouette_bool_dict.items()]
    intermediate_["valid"] = [value for key, value in filtered_cluster.items()]
    intermediate = pd.concat([intermediate, intermediate_])

    return intermediate

# Import Data
def CreateClusterHouseholds(households_building, results_file_path, run, WRITERESULTS):
    data = pd.read_csv(r"data/1_intermediate/Gneis/Gneis_data.csv", index_col=0)
    data = data.loc[:,data.sum(axis=0)<7200]
    data = data.T.drop_duplicates().T
    data = data.iloc[:,0:251]
    index = pd.date_range(start="01-01-2019 00:00", end="31-12-2019 23:30", freq="30min")
    data.index = pd.to_datetime(index, "%d-%m-%Y %H:%M")
    data.columns = [x.replace(" ", "") for x in data.columns]

    if households_building != None:
        data = data.loc[:, households_building]

    # Initiate algorithm
    layer = 0
    node = (0,)
    node_bool = True
    node_clusters = {node:0}
    node_households = {node:0}

    # Create array and dataframe
    array, dataframe = GenerateArray(data)
    if len(data.columns) <= 10:
        dataframe["Layer_0"] = 0
    else:
        node_households[node] = dataframe.count()[0]
        # Get all sil_score_dict and find the candidate number of clusters which is higher than 0.5 and it is the minimum value
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]

        # databases
        sil_score_dict = {key:0 for key in range_n_clusters}
        sample_silhouette_values_dict = {key:0 for key in range_n_clusters}
        sample_silhouette_bool_dict = {key:False for key in range_n_clusters}
        centers_dict = {key:0 for key in range_n_clusters}
        filtered_cluster = {key:False for key in range_n_clusters}

        n_cluster_candidate = 10
        for n_clusters in range_n_clusters:
            sil_score_dict[n_clusters], \
            sample_silhouette_values_dict[n_clusters], \
            centers_dict[n_clusters] = SilhouetteScoreRange(array=array, nClusters = n_clusters)
            bool_list = all(values >= 0 for values in sample_silhouette_values_dict)
            sample_silhouette_bool_dict[n_clusters] = bool_list
            #
            if sil_score_dict[n_clusters] >= 0.5 and bool_list == True:
                filtered_cluster[n_clusters] = True

                if n_cluster_candidate >= n_clusters: #update the candidate for clustering to always obtain the minimum
                    n_cluster_candidate = n_clusters

        if n_cluster_candidate == 10: #it implies that none value has pass the filter of 0.5 and bool_list == True
            n_cluster_candidate = max(sil_score_dict, key=sil_score_dict.get)

        node_clusters[node] = n_cluster_candidate # update the number of clusters in layer
        # for i in range(0, n_cluster_candidate):
        #     node_clusters[node + (i,)] = 0

        # Proceed to generate the cluster for layer
        dataKMeans, dataProcessed = KMeansAlgorithm(array=array, dataframe=dataframe, nClusters=n_cluster_candidate)

        # Update the dataframe
        dataframe.loc[dataKMeans.index, f"Layer_{layer}"] = dataKMeans.loc[:, "predicted cluster"]

        # Save databases for this layer
        list_tuples = [(node, i) for i in range_n_clusters]
        intermediate = pd.DataFrame(index = pd.MultiIndex.from_tuples(list_tuples),
                                    columns = ["sil_score", "sample_silhouette_bool", "valid"])

        intermediate["sil_score"] = [value for key, value in sil_score_dict.items()]
        intermediate["sample_silhouette_bool"] = [value for key, value in sample_silhouette_bool_dict.items()]
        intermediate["valid"] = [value for key, value in filtered_cluster.items()]
        # Layer for iteration
        stop = False
        while stop == False:
            layer += 1
            print("-"*10, "Layer ", layer, "-"*10)
            parent_node_list = [key for key in node_clusters.keys() if len(key)==layer]

            queries_dict = dict()
            for parent_node in parent_node_list:
                print("Querying for ", parent_node)
                for cluster in range(0, node_clusters[parent_node]):
                    tup = parent_node + (cluster,)
                    q = ' and '.join(f'{c} == {tup[i+1]}' for i, c in enumerate(dataframe.columns[2:]))
                    queries_dict[tup] = q

            for parent_node in parent_node_list:
                for cluster in range(0, node_clusters[parent_node]):
                    print("Clustering for ", parent_node, " and cluster ", cluster)
                    tup = parent_node + (cluster,)
                    filtered_dataframe = dataframe.query(queries_dict[tup]).copy()

                    bool_household = CheckNumberHouseholds(filtered_dataframe=filtered_dataframe, node_households=node_households, parent_node=parent_node, cluster=cluster)
                    intermediate= RunCluster(parent_node=parent_node,cluster=cluster, layer=layer,
                           dataframe=dataframe, filtered_dataframe=filtered_dataframe, range_n_clusters=range_n_clusters,
                           node_clusters=node_clusters, bool_household=bool_household, intermediate=intermediate, sil_score_dict = sil_score_dict,
                           sample_silhouette_values_dict = sample_silhouette_values_dict, centers_dict = centers_dict,
                                             sample_silhouette_bool_dict = sample_silhouette_bool_dict, filtered_cluster = filtered_cluster
                    )

            if np.isnan(dataframe[f"Layer_{layer}"].values).all():
                stop = True
            elif all(values == None for values in dataframe[f"Layer_{layer}"].values):
                stop = True
            else:
                stop = False

    if WRITERESULTS==True:
        print("Saving Clusters...")
        dataframe.to_csv(results_file_path + fr"/Clusters_{run}.csv")
        #intermediate.to_csv(r"data/1_intermediate/Gneis/ClusteringFromAlgorithm_IntermediateData.csv")
        #node_clusters_df = pd.DataFrame.from_dict(node_clusters, orient="index")
        #node_clusters_df.to_csv(r"data/1_intermediate/Gneis/ClusteringFromAlgorithm_NodeNumberClusters.csv")
        #node_households_df = pd.DataFrame.from_dict(node_households, orient="index")
        #node_households_df.to_csv(r"data/1_intermediate/Gneis/ClusteringFromAlgorithm_NumberHouseholdsCluster.csv")

    return dataframe

def ClustersToTuples(clusters_df, tab_file_path):
    input_sheet = clusters_df.replace(np.nan, 100)  # set nan to 100 to process the data
    input_sheet = input_sheet.drop(input_sheet.columns[:2].to_list(), axis=1)
    for c in input_sheet.columns:  # change to integers
        input_sheet[c] = input_sheet[c].astype("int")

    input_sheet_ = pd.DataFrame(index=input_sheet.index, columns=input_sheet.columns)
    for c in input_sheet_.columns:  # Create a new df that contains the cluster names in tuples
        input_sheet_[c] = input_sheet.loc[:, :c].apply(lambda row: ClustersTuples(row), axis=1)

    input_sheet_.to_csv(tab_file_path + f"/Sets_ClustersAggr.tab", header=True, sep="\t", mode="w")

    return input_sheet_


#%%
# layer += 1
# parent_node_list = [key for key in node_clusters.keys() if len(key)==layer]
#
# for parent_node in parent_node_list:
#     queries_dict = dict()
#     for cluster in range(0, node_clusters[parent_node]):
#         tup = parent_node + (cluster,)
#         q = ' and '.join(f'{c} == {tup[i+1]}' for i, c in enumerate(dataframe.columns[2:]))
#         queries_dict[tup] = q
#
#     for cluster in range(0, node_clusters[parent_node]):
#         tup = parent_node + (cluster,)
#         filtered_dataframe = dataframe.query(queries_dict[tup]).copy()
#
#         bool_household = CheckNumberHouseholds(filtered_dataframe=filtered_dataframe, node_households=node_households, parent_node=parent_node, cluster=cluster)
#         intermediate = RunCluster(parent_node=parent_node, cluster=cluster, layer=layer,
#                                   dataframe=dataframe, filtered_dataframe=filtered_dataframe, range_n_clusters=range_n_clusters, node_clusters=node_clusters, bool_household=bool_household,
#                                   intermediate=intermediate)
#
# #%% Layer 2
# layer = 2
# parent_node = (0,0)
# filtered_dataframe = dataframe[(dataframe["Layer_0"]==parent_node[0])&(dataframe["Layer_1"]==parent_node[1])]
#
# intermediate = RunCluster(parent_node=parent_node, layer=2, cluster=0,
#            dataframe=dataframe,filtered_dataframe=filtered_dataframe, range_n_clusters=range_n_clusters, node_clusters=node_clusters, node_households=node_households, intermediate=intermediate)
#
# parent_node = (0,1)
# filtered_dataframe = dataframe[(dataframe["Layer_0"]==parent_node[0])&(dataframe["Layer_1"]==parent_node[1])]
#
# intermediate = RunCluster(parent_node=parent_node, old_layer=1, layer=2,
#            dataframe=dataframe, range_n_clusters=range_n_clusters, node_clusters=node_clusters, node_households=node_households, intermediate=intermediate)
