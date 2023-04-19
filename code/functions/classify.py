from .utils import get_data, load_model


def get_prediction(test_index, name):
    # Load test data
    X_test, y_test = get_data(data_set='test')
    
    # Load model
    model = load_model(name)
    
    # Predict
    sample = X_test.loc[test_index]
    pred = model.predict(sample.values.reshape(1, -1))
    
    return pred[0]


def handle(event, context):
    test_index = int(event['pathParameters']['id'])
    pred = get_prediction(test_index, 'knn')
    return {
        'statusCode': 200,
        'body': f'Predicted {pred}\n'
    }
