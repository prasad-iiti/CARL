#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
RANDOM_SEED = 123


# In[3]:


rng = np.random.default_rng(RANDOM_SEED)


# In[4]:


assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6


# In[5]:


df = pd.read_csv("padufes20_metadata_official.csv")


# In[6]:


df.columns


# In[7]:


keep_cols = [
    "img_id",
    "diagnostic",
    "patient_id",
    "age",
    "gender",
    "region",
]
df = df[keep_cols]


# In[8]:


df["diagnostic"].value_counts(dropna="FALSE")


# In[9]:


df["dataset"] = "padufes20"


# In[10]:


df.rename(columns={
    "img_id": "image_name",
    "diagnostic": "label",
    "gender": "sex",
    "region": "location",
}, inplace=True)


# In[11]:


df["sex"] = df["sex"].fillna("unknown")


# In[12]:


df.isna().sum()


# In[13]:


assert df.isnull().sum().sum() == 0


# In[14]:


assert len(df) == df["image_name"].nunique()


# In[15]:


missing = []
for f in df["image_name"]:
    if not os.path.exists(os.path.join("/home/prasad/datasets/padufes20/images/", f)):
        missing.append(f)

print(len(missing))


# In[16]:


sorted(df["label"].unique().tolist())


# In[17]:


mapping = {
    "ACK": "AKIEC",
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NEV",
    "SCC": "SCC",
    "SEK": "SEK",
}


# In[18]:


assert set(df["label"].unique()) == set(mapping.keys())


# In[19]:


df["label"] = df["label"].map(mapping)


# In[20]:


assert df["label"].isna().sum() == 0


# In[21]:


assert sorted(df["label"].unique()) == [
    "AKIEC", "BCC", "MEL", "NEV", "SCC", "SEK"]


# In[22]:


df["label"].value_counts()


# In[23]:


print(df["label"].value_counts().sum())


# In[24]:


df["binary_label"]=(df["label"].isin(
    ["AKIEC", "MEL", "BCC", "SCC"])).astype(int)


# In[25]:


assert set(df["binary_label"].unique()) == {0,1}


# In[26]:


print(df["binary_label"].value_counts())


# In[27]:


print(df["binary_label"].value_counts().sum())


# In[28]:


print(pd.crosstab(df["binary_label"], df["label"]))


# In[29]:


df["sex"].value_counts()


# In[30]:


print(df["sex"].value_counts().sum())


# In[31]:


print(pd.crosstab(df["sex"], df["label"]))


# In[32]:


df["age"].value_counts()


# In[33]:


df["age"].value_counts().sum()


# In[34]:


df["age"].isna().sum()


# In[35]:


print(pd.crosstab(df["age"], df["label"]))


# In[36]:


df["location"].value_counts()


# In[37]:


df["location"].value_counts().sum()


# In[38]:


df["location"].isna().sum()


# In[39]:


print(pd.crosstab(df["location"], df["label"]))


# # Split

# In[40]:


len(df["patient_id"])


# In[41]:


patient_ids = df["patient_id"].dropna().unique().to_numpy()


# In[42]:


rng.shuffle(patient_ids)


# In[43]:


df["patient_id"].isna().sum()


# In[44]:


n_patients = len(patient_ids)
n_train = int(TRAIN_RATIO * n_patients)
n_val = int(VAL_RATIO * n_patients)


# In[45]:


train_patients = patient_ids[:n_train]
val_patients = patient_ids[n_train:n_train + n_val]
test_patients = patient_ids[n_train + n_val:]


# In[46]:


train_ids = set(train_patients)
val_ids   = set(val_patients)
test_ids  = set(test_patients)

print("Train ∩ Val :", train_ids & val_ids)
print("Train ∩ Test:", train_ids & test_ids)
print("Val ∩ Test  :", val_ids & test_ids)


# In[47]:


df["split"] = "unassigned"
df.loc[df["patient_id"].isin(train_patients), "split"] = "train"
df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
df.loc[df["patient_id"].isin(test_patients), "split"] = "test"


# In[48]:


assert (df["split"] != "unassigned").all()


# In[49]:


print("Patients per split:")
print(df.groupby("split")["patient_id"].nunique())


# In[50]:


print("\nImages per split:")
print(df["split"].value_counts())


# In[51]:


print("\nClass distribution per split:")
print(pd.crosstab(df["split"], df["label"]))


# In[52]:


print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


# In[53]:


pd.crosstab(df["age"], df["split"])


# In[54]:


pd.crosstab(df["sex"], df["split"])


# In[55]:


pd.crosstab(df["location"], df["split"])


# In[56]:


df.groupby(["split", "sex", "label"]).size()


# In[57]:


assert df["image_name"].is_unique
assert df["split"].isin(["train", "val", "test"]).all()
assert df["sex"].isin(["MALE", "FEMALE", "unknown"]).all()
assert df["label"].notna().all()


# In[58]:


leakage = (
    df.groupby("patient_id")["split"]
    .nunique()
    .gt(1)
    .sum()
)
print(f"Patients appearing in multiple splits: {leakage}")


# In[59]:


df=df[["dataset", "split", "image_name", "label", "binary_label", "patient_id",
        "age", "sex", "location"]]


# In[60]:


expected_cols = [
    "dataset",
    "split",
    "image_name",
    "label",
    "binary_label",
    "patient_id",
    "age",
    "sex",
    "location",
]
assert list(df.columns) == expected_cols


# In[61]:


print("=" * 60)
print("PADUFES20 metadata preprocessing completed")
print(f"Images   : {len(df)}")
print(f"Patients : {df['patient_id'].nunique()}")
print(f"Classes  : {df['label'].nunique()}")
print(df["split"].value_counts())
print("=" * 60)


# In[62]:


print(df.isnull().sum())


# In[63]:


assert df["image_name"].is_unique
assert df["patient_id"].notna().all()
assert df["label"].notna().all()
assert df["binary_label"].isin([0, 1]).all()
assert df["split"].isin(["train", "val", "test"]).all()


# In[64]:


assert df.isnull().sum().sum() == 0


# In[65]:


df.to_csv("padufes20_metadata_clean.csv", index=False, encoding="utf-8")


# In[66]:


df

