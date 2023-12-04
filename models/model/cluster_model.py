import umap
import hdbscan
class ClusterModel():
    def __init__(self, 
                 demension_reduction_methods: umap.UMAP,
                 clustering_method: hdbscan.HDBSCAN):
        self.demension_reduction_methods = demension_reduction_methods
        self.clustering_method = clustering_method

    def fit(self, embed_matrix):
        self.drm_object = self.demension_reduction_methods.fit(embed_matrix)
        self.drm_matrix = self.drm_object.transform(embed_matrix)
        self.cluster = self.clustering_method.fit(self.drm_matrix)