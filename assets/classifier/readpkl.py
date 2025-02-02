from joblib import load

with open('../../test/poging/fine_features.pkl', 'rb') as f:
    model = load(f)

print(model)
