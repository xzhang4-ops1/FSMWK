import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Union
import copy


class Clustering_base_class(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return self.name + '\n' + self.citation if hasattr(self, 'name') and hasattr(self, 'citation') else ''

    def _get_minkowski_centre(self, cluster_data: np.ndarray, p: float) -> np.ndarray:
        if p < 1.5:
            return np.median(cluster_data, axis=0)
        else:
            return np.mean(cluster_data, axis=0)


class Kmeans_base_class(Clustering_base_class):
    def __init__(self, k: int, initial_centroids: Union[str, np.ndarray] = 'random', max_iterations: int = 100) -> None:
        self.k = k
        self.labels = np.empty((0,))
        self.initial_centroids = initial_centroids.copy() if isinstance(initial_centroids,
                                                                        np.ndarray) else initial_centroids
        self.max_iterations = max_iterations

    def _update_centroids(self, p: float = 2.0) -> None:
        # Notice that if a cluster is lost, the clusters will be relabelled from 0 to new total
        unique_labels = np.unique(self.labels)
        self.centroids = np.array(
            [self._get_minkowski_centre(self.data[self.labels == k, :], p) for k in unique_labels])
        # If a cluster was lost, re-index (so self.labels is contiguous)
        if len(unique_labels) < self.labels.max() + 1:
            # Map the labels (i.e. if label 2 is gone, move the labels higher than 2 down)
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            # Reindex
            self.labels = np.array([label_map[label] for label in self.labels])

    def _initialise_centroids(self) -> None:
        if isinstance(self.initial_centroids, str):
            if self.initial_centroids == 'random':
                # Ensures the selected centroids are different
                unique_data = np.unique(self.data, axis=0)
                self.initial_centroids = unique_data[random.sample(range(unique_data.shape[0]), self.k), :]
                self.centroids = self.initial_centroids.copy()
                del unique_data
            elif self.initial_centroids == 'kmeans++':
                self._kmeans_plus_plus_init()
        elif isinstance(self.initial_centroids, np.ndarray) and self.initial_centroids.shape[0] == self.k:
            self.centroids = self.initial_centroids.copy()
        else:
            raise ValueError("Invalid value for initial_centroids parameter.")

    def _kmeans_plus_plus_init(self) -> None:
        self.centroids = [self.data[random.randint(0, self.data.shape[0] - 1)].copy()]
        for _ in range(1, self.k):
            distances = np.min([np.sum((self.data - c) ** 2, axis=1) for c in self.centroids], axis=0)
            probabilities = distances / distances.sum()
            self.centroids = np.vstack(
                (self.centroids, self.data[np.random.choice(range(self.data.shape[0]), p=probabilities)]))

    def _update_labels(self, distances_nxk: np.ndarray) -> bool:
        # Parameter
        # distances_nxk has shape (self.data.shape[0], self.k) so that distances_nxk[i,j] has the distance between self.data[i,:] and self.centroids[j,:]
        # Returns:
        # True if the clustering has changed
        # False if the clustering is the same from one of previosu 10 (to avoid cycles)

        # Prepare to store the history of labels so that it can detect cycles
        if not hasattr(self, '_label_history'):
            self._label_history = []
        # Add current set of labels to labels history
        self._label_history.append(self.labels)
        # Limit the size of the history to avoid memory issues, removes 1st row to keep 10
        if len(self._label_history) > 10:
            self._label_history.pop(0)

        self.distances = distances_nxk.copy()
        # Put inf instead of nan (in case of empty cluster)
        self.labels = np.where(np.isnan(self.distances), np.inf, self.distances).argmin(axis=1)
        # update self.distances so that it has the distance from each point to the nearest centroid
        self.distances = self.distances[np.arange(self.distances.shape[0]), self.labels].copy()

        # Check current label against the history of labels
        return not any(np.array_equal(self.labels, past_labels) for past_labels in self._label_history)

    # The below is just to make this class abstract as well
    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        pass
class Kmeans_weighted_base(Kmeans_base_class):
    def __init__(self, k: int, initial_centroids: Union[str, np.ndarray] = 'random',
                 initial_weights: Union[str, np.ndarray] = 'random', max_iterations: int = 100) -> None:
        super().__init__(k, initial_centroids, max_iterations)
        self.initial_weights = initial_weights.copy() if isinstance(initial_weights, np.ndarray) else initial_weights

    def _initialise_weights(self, n_features: int, n_weight_rows: int = 1) -> None:
        if isinstance(self.initial_weights, str):
            if self.initial_weights == 'equal':
                self.initial_weights = np.full((n_features,), 1 / n_features) if n_weight_rows == 1 else np.full(
                    (n_weight_rows, n_features), 1 / n_features)
            elif self.initial_weights == 'random':
                if n_weight_rows == 1:
                    self.initial_weights = np.random.rand(n_features)
                    self.initial_weights = self.initial_weights / self.initial_weights.sum()
                else:
                    self.initial_weights = np.random.rand(self.k, n_features)
                    self.initial_weights = self.initial_weights / self.initial_weights.sum(axis=1, keepdims=True)
            else:
                raise ValueError("Invalid initial_weights string value.")
        elif not isinstance(self.initial_weights, np.ndarray):
            raise ValueError("initial_weights should be a string or numpy array (np.ndarray).")
        elif (n_weight_rows == 1 and self.initial_weights.shape != (n_features,)) or (
                n_weight_rows != 1 and self.initial_weights.shape != (self.k, n_features)):
            raise ValueError("initial_weights is a numpy array but has the wrong shape.")
        # if nothing of the above applies, self.initial_weights is already a numpy array with correct shape
        self.weights = self.initial_weights.copy()

    @abstractmethod
    def _update_weights(self) -> None:
        pass

class MWKmeans(Kmeans_weighted_base):
    def __init__(self, k: int, p: float, initial_centroids: Union[str, np.ndarray] = 'random',
                 initial_weights: Union[str, np.ndarray] = 'equal', replications: int = 1,
                 max_iterations: int = 100) -> None:
        super().__init__(k, initial_centroids, initial_weights, max_iterations)
        self.p = p
        self.replications = replications
        self.name = 'Minkowski Weighted K-means'
        self.citation = '@article{de2012minkowski,\n title={Minkowski metric, feature weighting and anomalous cluster initializing in K-Means clustering},\n author={De Amorim, Renato Cordeiro and Mirkin, Boris},\n journal={Pattern Recognition},\n volume={45},\n number={3},\n pages={1061--1075},\n year={2012},\n publisher={Elsevier}\n}'

    def fit(self, data: np.ndarray) -> np.ndarray:
        self.data = data.copy()
        lowest_distances_sum = float('inf')
        for _ in range(self.replications):
            tmp_mwkmeans = copy.deepcopy(self)
            tmp_mwkmeans._fit_once()
            replication_distances_sum = tmp_mwkmeans.distances.sum()
            if replication_distances_sum < lowest_distances_sum:
                lowest_distances_sum = replication_distances_sum
                best_mwkmeans_run = copy.deepcopy(tmp_mwkmeans)
        self.__dict__.update(best_mwkmeans_run.__dict__)
        return self.labels, self.weights

    def _fit_once(self) -> None:
        self._initialise_centroids()
        self._initialise_weights(self.data.shape[1])
        self.iterations = 0
        while self.iterations < self.max_iterations and self._update_labels(
                np.sum(abs(self.data[:, np.newaxis, :] - self.centroids) ** self.p * (
                        self.weights[np.newaxis, :, :] ** self.p), axis=2)):
            self._update_centroids(self.p)
            self._update_weights()
            self.iterations += 1
        # self.distances = self.distances[np.arange(self.distances.shape[0]),self.labels].copy()

    def _update_weights(self) -> None:
        # calculate dispersion
        dispersion = np.zeros((self.centroids.shape[0], self.data.shape[1]))
        self.weights = dispersion.copy()
        for k in range(self.centroids.shape[0]):
            dispersion[k, :] = np.sum(abs(self.data[self.labels == k, :] - self.centroids[k, :]) ** self.p, axis=0)
        # We add 0.01 if iMWKmeans because the data may have only one point, leading to dispersion of zero with mean zero
        dispersion_mean = dispersion.mean()  # Calculates only once
        dispersion += 0.01 if isinstance(self,
                                         IMWKmeans._MWKmeans_update_first_centroid_only) or dispersion_mean == 0 else dispersion_mean

        # update actual weight
        if self.p != 1:
            normalized_dispersion = (dispersion[:, :, None] / dispersion[:, None, :]) ** (1 / (self.p - 1))
            sum_dispersion = normalized_dispersion.sum(axis=2)
            self.weights = 1 / sum_dispersion
        else:
            min_idx = np.argmin(dispersion, axis=1)
            self.weights[np.arange(self.weights.shape[0]), min_idx] = 1

    def _initialise_weights(self, n_features: int) -> None:
        self.weights = self.initial_weights.copy() if isinstance(self.initial_weights, np.ndarray) else np.full(
            (self.k, n_features), 1 / n_features)

    def _initialise_centroids(self) -> None:
        if isinstance(self.initial_centroids, str) and self.initial_centroids == 'mwkmeans++':
            self._mwkmeans_plus_plus_init()
        else:
            super()._initialise_centroids()

    def _mwkmeans_plus_plus_init(self) -> None:
        # Find the first centroid at random. Reshape for the c in self.centroids later.
        self.centroids = self.data[random.randint(0, self.data.shape[0] - 1)].copy().reshape((1, self.data.shape[1]))

        # Calculate dispersion in relation to the Minkowski centre. Add mean or 0.01 just in case
        centre = self._get_minkowski_centre(self.data, self.p).reshape(1, self.data.shape[1])
        dispersion = (abs(self.data - centre) ** self.p).sum(axis=0)
        dispersion_mean = dispersion.mean()
        dispersion += dispersion_mean if dispersion_mean > 0 else 0.01

        # Use the dispersion to find the initial weights. Add as many as there are centroids (hence tile)
        dispersion_exp = dispersion ** (1 / (self.p - 1))
        self.initial_weights = np.tile(1 / np.sum((dispersion_exp[:, None]) / dispersion_exp[None, :], axis=1),
                                       (self.k, 1))

        for _ in range(1, self.k):
            distances = np.min(
                [np.sum((abs(self.data - c) ** self.p) * (self.initial_weights[0] ** self.p), axis=1) for c in
                 self.centroids], axis=0)
            probabilities = distances / distances.sum()
            self.centroids = np.vstack(
                (self.centroids, self.data[np.random.choice(range(self.data.shape[0]), p=probabilities)]))


class IMWKmeans(MWKmeans):
    class _MWKmeans_update_first_centroid_only(MWKmeans):
        # This class is just used in IMWKmeans when finding the initial centroids and weights
        # It's a mwkmeans that updates only the tentative centroid
        def _update_centroids(self, p: float) -> None:
            # Update only the fist cluster (tentative_centroid in IMWKmeans)
            self.centroids[0, :] = self._get_minkowski_centre(self.data[self.labels == 0, :], p).reshape(
                (1, self.data.shape[1]))

        def fit(self, data: np.ndarray) -> np.ndarray:
            self.data = data.copy()
            self._initialise_centroids()
            self._initialise_weights(self.data.shape[1])
            self.iterations = 0
            # Distance calculation in _update_labels:
            # Uses only the weight from the first centroid (tentative_centroid in IMWKmeans)
            while self.iterations < self.max_iterations and self._update_labels(
                    np.sum(abs(self.data[:, np.newaxis, :] - self.centroids) ** self.p * (
                            self.weights[0].flatten() ** self.p), axis=2)):
                self._update_centroids(self.p)
                self._update_weights()
                self.iterations += 1
            return self.labels

    def __init__(self, k: int, p: float, threshold: int = 1, max_iterations: int = 100) -> None:
        # No initial centroids and no initial weights
        super().__init__(k, p, None, None, max_iterations)
        self.threshold = threshold
        self.name = 'Intelligent Minkowski Weighted K-means'

    def fit(self, data: np.ndarray) -> np.ndarray:
        self.data = data.copy()
        self.data_centre = self._get_minkowski_centre(self.data, self.p)
        self.initial_centroids = np.empty((0, self.data.shape[1]))
        self.initial_centroids_cardinalities = np.empty((0,))
        self.initial_weights = np.empty((0, self.data.shape[1]))

        while self.data.shape[0] > 0:
            # Tentative centroid is initialised to the farthest point from the centre
            # No actual need to weight the distance as the weights are all equal in the beginning
            tentative_centroid_idx = np.argmax((abs(self.data - self.data_centre) ** self.p).sum(axis=1))
            self.tentative_centroid = self.data[tentative_centroid_idx, :].copy()

            # Run MWKmeans updating only the tentative centroid
            mwkmeans_1_centroid = self._MWKmeans_update_first_centroid_only(2, self.p, np.vstack(
                (self.tentative_centroid, self.data_centre)), max_iterations=self.max_iterations)
            mwkmeans_1_centroid.fit(self.data)

            # If empty cluster, assign tentative centroid to cluster.
            if (mwkmeans_1_centroid.labels == 0).sum() == 0:
                mwkmeans_1_centroid.labels[tentative_centroid_idx] = 0
                self.tentative_centroid = self.data[tentative_centroid_idx, :].copy()
            else:
                self.tentative_centroid = mwkmeans_1_centroid.centroids[0, :]

                # If there are more data points in the anomalous cluster than the threshold, save
            tentative_centroid_cluster_cardinality = (mwkmeans_1_centroid.labels == 0).sum()
            if tentative_centroid_cluster_cardinality >= self.threshold:
                self.initial_centroids = np.vstack((self.initial_centroids, self.tentative_centroid))
                self.initial_weights = np.vstack((self.initial_weights, mwkmeans_1_centroid.weights[0]))
                self.initial_centroids_cardinalities = np.hstack(
                    (self.initial_centroids_cardinalities, tentative_centroid_cluster_cardinality))

            # Remove anomalous cluster from the data (copy to avoid shared memory issues)
            self.data = self.data[mwkmeans_1_centroid.labels != 0, :].copy()

        # If the number of cluster was stated and it's less than what imwkmeans found
        if isinstance(self.k, int) and self.k < self.initial_centroids.shape[0]:
            # The minus in the below is so that it's sorted descending.
            # Stable preserves the relative order of elements with equal keys
            idx = np.argsort(-self.initial_centroids_cardinalities, kind='stable')[:self.k]
            self.initial_centroids = self.initial_centroids[idx, :].copy()
            self.initial_centroids_cardinalities = self.initial_centroids_cardinalities[idx].copy()
            self.initial_weights = self.initial_weights[idx, :].copy()
        # Run the usual MWKmeans using the initial centroids and weights found
        self.mwkmeans = MWKmeans(self.initial_centroids.shape[0], self.p, self.initial_centroids, self.initial_weights,
                                 self.max_iterations)
        self.labels = self.mwkmeans.fit(data)
        # Assign mwkmeans results to be imwkmeans results
        self.centroids, self.weights, self.distances, self.iterations = self.mwkmeans.centroids, self.mwkmeans.weights, self.mwkmeans.distances, self.mwkmeans.iterations
        return self.labels