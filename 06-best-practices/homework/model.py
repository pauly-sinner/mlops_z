import pickle


def load_model(model_path = 'model.bin'):

    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr
