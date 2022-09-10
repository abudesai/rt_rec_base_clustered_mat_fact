Clustered Matrix Factorization implemented with Neural Network build in TensorFlow for Recommender - Base problem category as per Ready Tensor specifications.

- sklearn
- Tensorflow
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- recommender system

This algorithm differs from the traditional matrix factorization algorithm. We first use embedding of users and items to determine which of the K clusters they are most correlated with. Each of these user and item clusters themselves are D dimensional vectors (you may consider these as the more traditional factors in matrix factorization).

To summarize, users belong to clusters, each user cluster has a D-dimensional vector. So user => cluster => embedding in latent space. Similarly, item => item cluster => embedding in latent space.

We then concatenate the user and item latent vectors and pass through a dense layer with elu activation before using the final dense layer with single output representing the predicted rating.

The recommender is equipped with early stopping: the model would stop training if there is no significant improvement in a perdetermined number of epochs, with default equals 3.

The data preprocessing step includes indexing and standardization. Numerical values (ratings) are also scaled to [0,1] using min-max scaling.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as jester, anime, book-crossing, modcloth, amazon electronics, and movies.

This Recommender System is written using Python as its programming language. Tensorflow and ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
