import joblib

model_pak = joblib.load('model\\model_unistroi.pkl')
# model = model_pak["model"]
# label_encoder = model_pak['label_encoder']
print(model_pak)