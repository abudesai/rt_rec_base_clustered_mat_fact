
import numpy as np, pandas as pd
import os, sys
import warnings
from sklearn.utils import shuffle
import joblib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Layer, Softmax, Reshape, Dot, Add, Flatten, \
    Concatenate, Dense, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


MODEL_NAME = "recommender_base_clustered_matrix_factorizer"

model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"



COST_THRESHOLD = float('inf')



class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("\nCost is inf, so stopping training!!")
            self.model.stop_training = True


class ClusterEmbeddingLayer(Layer):
    def __init__(self, embedding_dim):
        super(ClusterEmbeddingLayer, self).__init__()
        self.D = embedding_dim
        
    def build(self, input_shape): 
        K = input_shape[-1]
        D = self.D
        self.u_cluster_factors = tf.Variable(initial_value=np.random.randn(1, D, K), dtype=tf.float32, trainable=True)  # (1, D, K)
        self.u_cluster_bias = tf.Variable(initial_value=np.zeros(shape=(1, 1, K)), dtype=tf.float32, trainable=True)  # (1, 1, K)
        
        
    def call(self, inputs): 
        factors = tf.math.multiply(inputs, tf.math.add(self.u_cluster_factors, self.u_cluster_bias)) # (N, D, K)
        factors = tf.math.reduce_sum(factors, axis=2)  # (N, D)
        return factors
        

class Recommender():

    def __init__(self, N, M, K=10, D=10, lr = 1e-3,  batch_size=256, **kwargs  ):
        '''
        N: num of users
        M: num of items
        K: num of clusters (same for both number of user clusters, and num of item clusters)
        D: dimensionality of each cluster 
        '''
        self.N = N
        self.M = M
        self.K = K
        self.D = D
        self.lr = lr
        self.batch_size = batch_size

        self.model = self.build_model()
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.lr),
            metrics=['mae'],
        )            
       

    def build_model(self): 
        N, M, K, D = self.N, self.M, self.K, self.D
        
        # keras model
        u = Input(shape=(1,))
        m = Input(shape=(1,))
        
        u_cluster_membership = Embedding(N, K)(u) # (N, 1, K)   
        u_factors = ClusterEmbeddingLayer(embedding_dim=D)(u_cluster_membership)   # (N, D)        
        
        i_cluster_membership = Embedding(M, K)(m) # (N, 1, K)           
        i_factors = ClusterEmbeddingLayer(embedding_dim=D)(i_cluster_membership)   # (N, D) 
        
        out = Concatenate()([u_factors, i_factors]) # (N, 2D)
        out = Dense(10, activation='elu')(out)
        out = Dense(1)(out)
        
        outputs = out 
        
        

        model = Model(inputs=[u, m], outputs=outputs)
        return model


    def fit(self, X, y, validation_split=None, epochs=100, verbose=0): 
                
        early_stop_loss = 'val_loss' if validation_split is not None else 'loss'
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-4, patience=3) 
        infcost_stop_callback = InfCostStopCallback()

        history = self.model.fit(
                x = [ X[:, 0], X[:, 1] ],
                y = y, 
                validation_split = validation_split,
                batch_size = self.batch_size,
                epochs=epochs,
                verbose=verbose,
                shuffle=True,
                callbacks=[early_stop_callback, infcost_stop_callback]
            )
        return history


    def predict(self, X): 
        preds = self.model.predict([ X[:, 0], X[:, 1] ], verbose=1)
        return preds 

    def summary(self):
        self.model.summary()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        return self.model.evaluate(
                x = [ x_test[:, 0], x_test[:, 1] ],
                y = y_test, 
                verbose=0)   
                
        

    def save(self, model_path): 
        model_params = {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))

        self.model.save_weights(os.path.join(model_path, model_wts_fname))


    @staticmethod
    def load(model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        mf = Recommender(**model_params)
        mf.model.load_weights(os.path.join(model_path, model_wts_fname)).expect_partial()
        return mf


def get_data_based_model_params(X): 
    '''
    returns a dictionary with N: number of users and M = number of items
    This assumes that the given numpy array (X) has users by id in first column, 
    and items by id in 2nd column. the ids must be 0 to N-1 and 0 to M-1 for users and items.
    '''
    N = int(X[:, 0].max()+1)
    M = int(X[:, 1].max()+1)
    return {"N":N, "M": M}



def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = Recommender.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f, indent=2)


if __name__ == "__main__": 
    N = 5   # num users
    M = 3   # num items
    D = 6   # embedding dim
    K = 4   # num clusters
    num_samples = 20
    
    users = np.random.randint(N, size=num_samples).reshape(-1,1)
    items = np.random.randint(M, size=num_samples).reshape(-1,1)
    ratings = np.random.randn(num_samples).reshape(-1,1)
    
    R = np.concatenate([users, items, ratings], axis=1)
    # print(R.shape)
    
    model = Recommender(
        N=N,
        M=M,
        K=K,
        D=D
    )
    
    preds = model.predict(R)
    
    print(f"{num_samples=}, {N=}, {M=}, {K=}, {D=}")
    print(preds.shape)
    # print(preds[:2])
    