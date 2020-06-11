from keras.models import load_model


def evaluate(model_path, x_test, y_test):
    model = load_model(model_path)
    return model.evaluate(x_test, y_test)
