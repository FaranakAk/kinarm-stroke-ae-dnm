
# libraries
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Utilities
import os
import csv
from time import time
import matplotlib.pyplot as plt

# Tensorflow/Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Dataset helper function
#from datasets import load_data

# DESOM components
from SOM import SOMLayer
from AE import mlp_autoencoder
from metrics import *

import tensorflow.keras.backend as K

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping

##########################################################

def som_loss(weights, distances):
    """
    SOM loss

    # Arguments
        weights: weights for the weighted sum, Tensor with shape `(n_samples, n_prototypes)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, Tensor with shape `(n_samples, n_prototypes)`
    # Return
        SOM reconstruction loss
    """
    loss_1 = tf.reduce_mean(tf.reduce_sum(weights*distances, axis=1))
    q = 1/(1+distances)
    q = q/tf.expand_dims(tf.reduce_sum(q, axis=1), axis=1)
    p = q**2 * tf.reduce_sum(q, axis=0)
    p = p/tf.expand_dims(tf.reduce_sum(p, axis=1), axis=1)
    
#    p = K.clip(p, K.epsilon(), 1)
#    q = K.clip(q, K.epsilon(), 1)
    loss_2 = K.sum(p * (K.log(p / q)/K.log(tf.convert_to_tensor(float(10)))), axis=-1)
    loss_2 = tf.reduce_mean(loss_2)
    beta = 1
    return loss_1+beta*loss_2


def ae_loss(layer): 
    def loss(y_true, y_pred):
        
        rec_loss = tf.keras.losses.MSE(y_pred, y_true)
        lmbd = 1e-6
        par_loss = 0
        for l in layer:
            par_loss = par_loss + K.sum(l.weights[0]**2)

        return rec_loss+lmbd*par_loss
    
    return loss


def kmeans_loss(y_pred, distances):
    """
    k-means reconstruction loss

    # Arguments
        y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        distances: pairwise squared euclidean distances between inputs and prototype vectors, numpy.array with shape `(n_samples, n_prototypes)`
    # Return
        k-means reconstruction loss
    """
    return np.mean([distances[i, y_pred[i]] for i in range(len(y_pred))])


class DESOM:
    """
    Deep Embedded Self-Organizing Map (DESOM) model

    # Example
        ```
        desom = DESOM(encoder_dims=[784, 500, 500, 2000, 10], map_size=(10,10))
        ```

    # Arguments
        encoder_dims: list of numbers of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer (latent dim)
        map_size: tuple representing the size of the rectangular map. Number of prototypes is map_size[0]*map_size[1]
    """

    def __init__(self, encoder_dims, map_size):
        self.encoder_dims = encoder_dims
        self.input_dim = self.encoder_dims[0]
        self.map_size = map_size
        self.n_prototypes = map_size[0]*map_size[1]
        self.pretrained = False
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.model = None
    
    def initialize(self, ae_act='relu', ae_init='glorot_uniform'):
        """
        Create DESOM model

        # Arguments
            ae_act: activation for AE intermediate layers
            ae_init: initialization of AE layers
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = mlp_autoencoder(self.encoder_dims, ae_act, ae_init)
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
        # Create DESOM model
        self.model = Model(inputs=self.autoencoder.input,
                            outputs=[self.autoencoder.output, som_layer])
    
    @property
    def prototypes(self):
        """
        Returns SOM code vectors
        """
        return self.model.get_layer(name='SOM').get_weights()[0]

    def compile(self, gamma, gamma1, optimizer):
        """
        Compile DESOM model

        # Arguments
            gamma: coefficient of SOM loss
            optimizer: optimization algorithm
        """
        self.model.compile(loss={'decoder_0': ae_loss([self.model.get_layer(index=x) for x in range(1,len(self.encoder_dims))]), 'SOM': som_loss},
                            loss_weights=[gamma1, gamma],
                            optimizer=optimizer)
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DESOM model

        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE

        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def init_som_weights(self, X):
        """
        Initialize with a sample w/o remplacement of encoded data points.

        # Arguments
            X: numpy array containing training set or batch
        """
        sample = X[np.random.choice(X.shape[0], size=self.n_prototypes, replace=False)]
        encoded_sample = self.encode(sample)
        self.model.get_layer(name='SOM').set_weights([encoded_sample])

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer

        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)
    
    def decode(self, x):
        """
        Decoding function. Decodes encoded features from latent space

        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)

    def predict(self, x):
        """
        Predict best-matching unit using the output of SOM layer

        # Arguments
            x: data point
        # Return
            index of the best-matching unit
        """
        _, d = self.model.predict(x, verbose=0)
        return d.argmin(axis=1), d.min(axis=1)

    def map_dist(self, y_pred):
        """
        Calculate pairwise Manhattan distances between cluster assignments and map prototypes (rectangular grid topology)
        
        # Arguments
            y_pred: cluster assignments, numpy.array with shape `(n_samples,)`
        # Return
            pairwise distance matrix (map_dist[i,k] is the Manhattan distance on the map between assigned cell of data point i and cell k)
        """
#        labels = np.arange(self.n_prototypes)
#        tmp = np.expand_dims(y_pred, axis=1)
#        d_row = np.abs(tmp-labels) // self.map_size[1]
#        d_col = np.abs(tmp % self.map_size[1] - labels % self.map_size[1])
        
        bmu_row, bmu_col = y_pred // self.map_size[1], y_pred % self.map_size[1]
        d_row, d_col = [], []
        for i in range(self.n_prototypes):
            x_row, x_col = i // self.map_size[1], i % self.map_size[1]
            d_row.append(np.abs(bmu_row-x_row))
            d_col.append(np.abs(bmu_col-x_col))
        d_row = np.array(d_row).T    
        d_col = np.array(d_col).T
#        return d_row + d_col #manhattan distance
        return np.sqrt(d_row**2+d_col**2)



    

    @staticmethod
    def neighborhood_function(d, T, neighborhood='gaussian'):
        """
        SOM neighborhood function (gaussian neighborhood)

        # Arguments
            x: distance on the map
            T: temperature parameter
        # Return
            neighborhood weight
        """
        if neighborhood == 'gaussian':
            return np.exp(-(d ** 2) / (T ** 2))
        elif neighborhood == 'window':
            return (d <= T).astype(np.float32)
    
    def pretrain(self, X, V,
                    optimizer='adam',
                    epochs=200,
                    batch_size=256,
                    save_dir='results/tmp'):
        """
        Pre-train the autoencoder using only MSE reconstruction loss
        Saves weights in h5 format.

        # Arguments
            X: training set
            optimizer: optimization algorithm
            epochs: number of pre-training epochs
            batch_size: training batch size
            save_dir: path to existing directory where weights will be saved
        """
        print('Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        # Begin pretraining
        t0 = time()
        history = self.autoencoder.fit(X, X, validation_data=(V,V),batch_size=batch_size, shuffle=1, epochs=epochs, verbose=1,
                                    callbacks = [EarlyStopping( monitor='val_loss', mode='min',
                                    patience=10,restore_best_weights=True)])
    
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.title('Autoencoder alone loss')
        plt.show()    
    
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights('{}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        print('Pretrained weights are saved to {}/ae_weights-epoch{}.h5'.format(save_dir, epochs))
        self.pretrained = True
    
    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            iterations=10000,
            som_iterations=10000,
            eval_interval=10,
            save_epochs=5,
            batch_size=256,
            Tmax=10,
            Tmin=0.1,
            decay='exponential',
            save_dir='results/tmp',
            patience=30,
            delta=0.0001,
            bestModel_saveName=''):
        """
        Training procedure

        # Arguments
            X_train: training set
            y_train: (optional) training labels
            X_val: (optional) validation set
            y_val: (optional) validation labels
            iterations: number of training iterations
            som_iterations: number of iterations where SOM neighborhood is decreased
            eval_interval: evaluate metrics on training/validation batch every eval_interval iterations
            save_epochs: save model weights every save_epochs epochs
            batch_size: training batch size
            Tmax: initial temperature parameter
            Tmin: final temperature parameter
            decay: type of temperature decay ('exponential' or 'linear')
            save_dir: path to existing directory where weights and logs are saved
        """
        if not self.pretrained:
            print('Autoencoder was not pre-trained!')

        save_interval = X_train.shape[0] // batch_size * save_epochs # save every save_epochs epochs
        print('Save interval:', save_interval)

        # Logging file
        logfile = open(save_dir + '/desom_log.csv', 'w')
        fieldnames = ['iter', 'T', 'L', 'Lr', 'Lsom', 'Lkm', 'Ltop', 'quantization_err', 'topographic_err', 'latent_quantization_err', 'latent_topographic_err']
        if X_val is not None:
            fieldnames += ['L_val', 'Lr_val', 'Lsom_val', 'Lkm_val', 'Ltop_val', 'quantization_err_val', 'topographic_err_val', 'latent_quantization_err_val', 'latent_topographic_err_val']
        if y_train is not None:
            fieldnames += ['acc', 'pur', 'nmi', 'ari']
        if y_val is not None:
            fieldnames += ['acc_val', 'pur_val', 'nmi_val', 'ari_val']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        # Set and compute some initial values
        index = 0
        if X_val is not None:
            index_val = 0

        train_loss_hist = []
        val_loss_hist = []
        val_loss_min = -np.Inf
        best_score = 0
        counter = 0
        early_stop = False
        
        
        for ite in range(iterations):
            
            if early_stop:
                print('Early stopping!')
                best_model.save_weights(save_dir + '/DESOM_bestModel_' + bestModel_saveName + '.h5')
                return train_loss_hist, val_loss_hist
            
            shuff_ind = np.random.permutation(len(X_train))
            X_train = X_train[shuff_ind]
            y_train = y_train[shuff_ind]
            shuff_ind = np.random.permutation(len(X_val))
            X_val = X_val[shuff_ind]
            y_val = y_val[shuff_ind]
            # Get training and validation batches
            
            X_batch = X_train
            if y_train is not None:
                y_batch = y_train

            if X_val is not None:
                X_val_batch = X_val
                if y_val is not None:
                    y_val_batch = y_val
                
            # Compute cluster assignments for batches
            _, d = self.model.predict(X_batch)
            y_pred = d.argmin(axis=1)
            if X_val is not None:
                _, d_val = self.model.predict(X_val_batch)
                y_val_pred = d_val.argmin(axis=1)

            # Update temperature parameter
            if ite < som_iterations:
                if decay == 'exponential':
                    T = Tmax*(Tmin/Tmax)**(ite/(som_iterations-1))
                elif decay == 'linear':
                    T = Tmax - (Tmax-Tmin)*(ite/(som_iterations-1))
            
            # Compute topographic weights batches
            w_batch = self.neighborhood_function(self.map_dist(y_pred), T)
            if X_val is not None:
                w_val_batch = self.neighborhood_function(self.map_dist(y_val_pred), T)

            # Train on batch

            if X_val is not None:
                loss = self.model.fit(X_batch, [X_batch, w_batch], 
                                        validation_data=(X_val_batch, [X_val_batch, w_val_batch]),
                                        verbose=0, epochs=1, batch_size=batch_size, shuffle=0) #model.metrics_names will give you the display labels for the scalar outputs.
                train_loss_hist.append(loss.history['loss'])
                val_loss_hist.append(loss.history['val_loss'])
            else:
                loss = self.model.fit(X_batch, [X_batch, w_batch], 
                                        verbose=0, epochs=1, batch_size=batch_size, shuffle=0) #model.metrics_names will give you the display labels for the scalar outputs.
                train_loss_hist.append(loss.history['loss'])
                
        
#            return loss
#            if ite % eval_interval == 0:
            # Initialize log dictionary
            logdict = dict(iter=ite, T=T)

            # Get SOM weights and decode to original space
            decoded_prototypes = self.decode(self.prototypes)

            # Evaluate losses and metrics
            
            print('iteration {} - T={}'.format(ite, T))
            logdict['L'] = loss.history['loss'][0]
            logdict['Lr'] = loss.history['decoder_0_loss'][0]
            logdict['Lsom'] = loss.history['SOM_loss'][0]
            logdict['Lkm'] = kmeans_loss(y_pred, d)
            logdict['Ltop'] = loss.history['SOM_loss'][0] - logdict['Lkm']
            logdict['latent_quantization_err'] = quantization_error(d)
            logdict['latent_topographic_err'] = topographic_error(d, self.map_size)
            d_original = np.square((np.expand_dims(X_train, axis=1) - decoded_prototypes)).sum(axis=2)
            logdict['quantization_err'] = quantization_error(d_original)
            logdict['topographic_err'] = topographic_error(d_original, self.map_size)
#            print('[Train] - Lr={:f}, Lsom={:f} (Lkm={:f}/Ltop={:f}) - total loss={:f}'.format(logdict['Lr'], logdict['Lsom'], logdict['Lkm'], logdict['Ltop'], logdict['L']))
#            print('[Train] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err'], logdict['topographic_err']))
            if X_val is not None:
#                    val_loss = self.model.test_on_batch(X_val_batch, [X_val_batch, w_val_batch])
                logdict['L_val'] = loss.history['val_loss'][0]
                logdict['Lr_val'] = loss.history['val_decoder_0_loss'][0]
                logdict['Lsom_val'] = loss.history['val_SOM_loss'][0]
                logdict['Lkm_val'] = kmeans_loss(y_val_pred, d_val)
                logdict['Ltop_val'] = loss.history['val_SOM_loss'][0] - logdict['Lkm_val']
                logdict['latent_quantization_err_val'] = quantization_error(d_val)
                logdict['latent_topographic_err_val'] = topographic_error(d_val, self.map_size)
                d_original_val = np.square((np.expand_dims(X_val, axis=1) - decoded_prototypes)).sum(axis=2)
                logdict['quantization_err_val'] = quantization_error(d_original_val)
                logdict['topographic_err_val'] = topographic_error(d_original_val, self.map_size)   
#                print('[Val] - Lr={:f}, Lsom={:f} (Lkm={:f}/Ltop={:f}) - total loss={:f}'.format(logdict['Lr_val'], logdict['Lsom_val'], logdict['Lkm_val'], logdict['Ltop_val'], logdict['L_val']))
#                print('[Val] - Quantization err={:f} / Topographic err={:f}'.format(logdict['quantization_err_val'], logdict['topographic_err_val']))

            # Evaluate the clustering performance using labels
            if y_train is not None:
                logdict['acc'] = cluster_acc(y_train, y_pred)
                logdict['pur'] = cluster_purity(y_train, y_pred)
                logdict['nmi'] = metrics.normalized_mutual_info_score(y_train, y_pred)
                logdict['ari'] = metrics.adjusted_rand_score(y_train, y_pred)
#                print('[Train] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc'], logdict['pur'], logdict['nmi'], logdict['ari']))
            if y_val is not None:
                logdict['acc_val'] = cluster_acc(y_val, y_val_pred)
                logdict['pur_val'] = cluster_purity(y_val, y_val_pred)
                logdict['nmi_val'] = metrics.normalized_mutual_info_score(y_val, y_val_pred)
                logdict['ari_val'] = metrics.adjusted_rand_score(y_val, y_val_pred)
#                print('[Val] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc_val'], logdict['pur_val'], logdict['nmi_val'], logdict['ari_val']))
                
            logwriter.writerow(logdict)

            # Save intermediate model
            if ite % save_interval == 0:
#                     self.model.save_weights(save_dir + '/DESOM_model_' + str(ite) + '.h5')
                    print('Saved model to:', save_dir + '/DESOM_model_' + str(ite) + '.h5')
                    
            # Early stopping

            my_val_loss = loss.history['val_loss'][0]
            score = -loss.history['val_loss'][0]
            
            if best_score==0:
                best_score = score
                print(f'Validation loss decreased ({val_loss_min:.6f} --> {my_val_loss:.6f}).  Saving model ...')
                try:
                    os.remove(save_dir + '/DESOM_bestModel_' + bestModel_saveName + '.h5')
                except:
                    pass
                # self.model.save_weights(save_dir + '/DESOM_bestModel_'+ bestModel_saveName + '.h5')
                best_model = self.model
                val_loss_min = loss.history['val_loss'][0]
            elif score < best_score + delta:
                counter += 1
                print(f'EarlyStopping counter: {counter} out of {patience}')
                if counter >= patience:
                    early_stop = True
            else:
                best_score = score
                print(f'Validation loss decreased ({val_loss_min:.6f} --> {my_val_loss:.6f}).  Saving model ...')
                try:
                    os.remove(save_dir + '/DESOM_bestModel_' + bestModel_saveName+ '.h5')
                except:
                    pass
                # self.model.save_weights(save_dir + '/DESOM_bestModel_' + bestModel_saveName + '.h5')
                best_model = self.model
                val_loss_min = loss.history['val_loss'][0]
                counter = 0
            print('-----------------------------')

        # Save the final model
        logfile.close()
        # print('saving model to:', save_dir + '/DESOM_model_final.h5')
        best_model.save_weights(save_dir + '/DESOM_bestModel_' + bestModel_saveName + '.h5')
        return train_loss_hist, val_loss_hist

##########################################################
import numpy as np
from sklearn.preprocessing import LabelEncoder
DIAMETER_METHODS = ['farthest']
CLUSTER_DISTANCE_METHODS = [ 'farthest']

def inter_cluster_distances(labels, distances, method='nearest'):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculates cluster diameters
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, distances, diameter_method='farthest',
         cdist_method='nearest'):
    """
    Dunn index for cluster validation (larger is better).
    
    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace
    
    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)` is the diameter of cluster :math:`c_k`.
    Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their closest elements. Cluster diameter can be defined as the mean distance between all elements in the cluster, between all elements to the cluster centroid, or as the distance between the two furthest elements.
    The higher the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart (large :math:`d \\left( c_i,c_j \\right)`).
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param diameter_method: see :py:function:`diameter` `method` parameter
    :param cdist_method: see :py:function:`diameter` `method` parameter
    
    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    if len( ic_distances.nonzero()[0] )!=0:
        min_distance = min(ic_distances[ic_distances.nonzero()])
    else:
        min_distance = 0
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter

