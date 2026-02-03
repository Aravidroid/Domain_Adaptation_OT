#data/split.py
import scipy.io
import pandas as pd

# Load the .mat file
data = scipy.io.loadmat("MNIST_vs_USPS.mat")

# Extract source and target
Xs = data['X_src'].T   # <-- TRANSPOSE HERE
ys = data['Y_src'].ravel()

Xt = data['X_tar'].T   # <-- TRANSPOSE HERE
yt = data['Y_tar'].ravel()

# Convert to DataFrame
source_df = pd.DataFrame(Xs)
source_df['label'] = ys

target_df = pd.DataFrame(Xt)
target_df['label'] = yt

# Save CSV files
source_df.to_csv("source.csv", index=False)
target_df.to_csv("target.csv", index=False)

print("✅ source.csv and target.csv created successfully")
