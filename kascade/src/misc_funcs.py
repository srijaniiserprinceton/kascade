import pickle

def save_ckpt(name, **arrays):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(arrays, f)