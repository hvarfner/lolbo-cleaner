import pandas as pd

df = pd. read_csv("/home/hvarfner/Documents/lolbo-cleaner/selfies_vae/guacamol_train_data_first_20k.csv")

for col in df.columns:
    if (col == "selfie") or (col == "smile"):
        continue
    new_df = df.loc[:, ["selfie", col]]
    new_df.columns = ["x", "y"]
    new_df.to_csv(f"/home/hvarfner/Documents/lolbo-cleaner/selfies_vae/init/init_{col}.csv")
    