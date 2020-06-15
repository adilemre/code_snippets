def find_optimal_n_clusters(input_data_frame,input_clustering_feature,input_cluster_upper_threshold):
    from sklearn.cluster import KMeans
    import seaborn as sns
    import matplotlib.pyplot as plt
    X=input_data_frame[input_clustering_feature].values.reshape(-1,1)
    Cluster_results_df=pd.DataFrame(columns=['n_clusters','interia','cluster_sizes'])
    for n_cluster in np.arange(2,input_cluster_upper_threshold):
        Clustering_KMeans=KMeans(n_clusters=n_cluster,
                                 random_state=15).fit(X)
        labels=list(Clustering_KMeans.labels_)
        cluster_sizes=[labels.count(x) for x in set(labels)]
        inertia=Clustering_KMeans.inertia_
        Cluster_results_df.loc[-1]=[n_cluster,inertia,cluster_sizes]
        Cluster_results_df.index=Cluster_results_df.index+1
    plt.figure(figsize=(12,6))
    sns.lineplot(x='n_clusters',y='interia',data=Cluster_results_df)
    plt.xticks(np.arange(2, input_cluster_upper_threshold, step=1));
