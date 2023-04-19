from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .utils import get_data, save_model


def preprocessing(X):
    return (X - 127.5) / 127.5


def handle(event, context):
    # Load data
    X_train, y_train = get_data()
    
    # Define model
    k = 5
    pipe = Pipeline([('scaler', FunctionTransformer(preprocessing)),
                     ('clf', KNeighborsClassifier(n_neighbors=k, weights='distance'))])
    
    # Train
    pipe.fit(X_train, y_train)
    
    # Save model
    save_model(pipe, 'knn')
    
    print(f'Trained model with k={k}')
