import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the data
with open("univ_pert_reg.pkl", "rb") as f:
    data = pickle.load(f)
data = data.squeeze(0).permute(1, 2, 0).cpu().numpy()
data = (data - data.min()) / (data.max() - data.min())
plt.imshow(data)
plt.savefig('univ_pert_reg.png')