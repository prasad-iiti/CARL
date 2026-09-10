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


df = pd.read_csv("HAM10000_metadata_official.csv")


# In[6]:


df.drop(columns=["dx_type", "dataset"], inplace=True)


# In[7]:


df["dataset"] = "ham10000"


# In[8]:


df.rename(columns={
    "image_id": "image_name",
    "dx": "label",
    "lesion_id": "patient_id",
    "localization": "location",
}, inplace=True)


# In[9]:


df["age"] = df["age"].fillna("unknown")


# In[10]:


df.isna().sum()


# In[11]:


assert df.isnull().sum().sum() == 0


# In[12]:


df["image_name"] = df["image_name"].apply(
    lambda x: x if str(x).endswith(".jpg") else str(x) + ".jpg")


# In[13]:


assert len(df) == df["image_name"].nunique()


# In[14]:


missing = []
for f in df["image_name"]:
    if not os.path.exists(os.path.join("/home/prasad/datasets/ham10000/images/", f)):
        missing.append(f)

print(len(missing))


# In[15]:


mapping = {
    "akiec": "AKIEC",
    "bcc"  : "BCC",
    "bkl"  : "BKL",
    "df"   : "DF",
    "mel"  : "MEL",
    "nv"   : "NEV",
    "vasc" : "VASC",
}


# In[16]:


assert set(df["label"].unique()) == set(mapping.keys())


# In[17]:


df["label"] = df["label"].map(mapping)


# In[18]:


assert df["label"].isna().sum() == 0


# In[19]:


assert sorted(df["label"].unique()) == [
    "AKIEC", "BCC", "BKL", "DF", "MEL", "NEV", "VASC"]


# In[20]:


df["label"].value_counts()


# In[21]:


print(df["label"].value_counts().sum())


# In[22]:


assert df["label"].isna().sum() == 0


# In[23]:


df["binary_label"]=(df["label"].isin(
    ["MEL", "BCC", "AKIEC"])).astype(int)


# In[24]:


assert set(df["binary_label"].unique()) == {0,1}


# In[25]:


print(df["binary_label"].value_counts())


# In[26]:


print(df["binary_label"].value_counts().sum())


# In[27]:


print(pd.crosstab(df["binary_label"], df["label"]))


# In[28]:


df["sex"].value_counts()


# In[29]:


print(df["sex"].value_counts().sum())


# In[30]:


print(pd.crosstab(df["sex"], df["label"]))


# In[31]:


df["age"].value_counts()


# In[32]:


df["age"].value_counts().sum()


# In[33]:


df["age"].isna().sum()


# In[34]:


print(pd.crosstab(df["age"], df["label"]))


# In[35]:


df["location"].value_counts()


# In[36]:


df["location"].value_counts().sum()


# In[37]:


df["location"].isna().sum()


# In[38]:


print(pd.crosstab(df["location"], df["label"]))


# # Split

# In[39]:


patient_ids = df["patient_id"].dropna().unique().to_numpy()


# In[40]:


rng.shuffle(patient_ids)


# In[41]:


df["patient_id"].isna().sum()


# In[42]:


n_patients = len(patient_ids)
n_train = int(TRAIN_RATIO * n_patients)
n_val = int(VAL_RATIO * n_patients)


# In[43]:


train_patients = patient_ids[:n_train]
val_patients = patient_ids[n_train:n_train + n_val]
test_patients = patient_ids[n_train + n_val:]


# In[44]:


train_ids = set(train_patients)
val_ids   = set(val_patients)
test_ids  = set(test_patients)

print("Train ∩ Val :", train_ids & val_ids)
print("Train ∩ Test:", train_ids & test_ids)
print("Val ∩ Test  :", val_ids & test_ids)


# In[45]:


df["split"] = "unassigned"
df.loc[df["patient_id"].isin(train_patients), "split"] = "train"
df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
df.loc[df["patient_id"].isin(test_patients), "split"] = "test"


# In[46]:


assert (df["split"] != "unassigned").all()


# In[47]:


print("Patients per split:")
print(df.groupby("split")["patient_id"].nunique())


# In[48]:


print("\nImages per split:")
print(df["split"].value_counts())


# In[49]:


print("\nClass distribution per split:")
print(pd.crosstab(df["split"], df["label"]))


# In[50]:


print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


# In[51]:


pd.crosstab(df["age"], df["split"])


# In[52]:


pd.crosstab(df["sex"], df["split"])


# In[53]:


pd.crosstab(df["location"], df["split"])


# In[54]:


df.groupby(["split", "sex", "label"]).size()


# In[55]:


assert df["image_name"].is_unique
assert df["split"].isin(["train", "val", "test"]).all()
assert df["sex"].isin(["male", "female", "unknown"]).all()
assert df["label"].notna().all()


# In[56]:


leakage = (
    df.groupby("patient_id")["split"]
    .nunique()
    .gt(1)
    .sum()
)
print(f"Patients appearing in multiple splits: {leakage}")


# In[57]:


df=df[["dataset", "split", "image_name", "label", "binary_label", "patient_id",
        "age", "sex", "location"]]


# In[58]:


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


# In[59]:


print("=" * 60)
print("HAM10000 metadata preprocessing completed")
print(f"Images   : {len(df)}")
print(f"Patients : {df['patient_id'].nunique()}")
print(f"Classes  : {df['label'].nunique()}")
print(df["split"].value_counts())
print("=" * 60)


# In[60]:


print(df.isnull().sum())


# In[61]:


assert df["image_name"].is_unique
assert df["patient_id"].notna().all()
assert df["label"].notna().all()
assert df["binary_label"].isin([0, 1]).all()
assert df["split"].isin(["train", "val", "test"]).all()


# In[62]:


assert df.isnull().sum().sum() == 0


# In[63]:


df.to_csv("ham10000_metadata_clean.csv", index=False, encoding="utf-8")


# In[64]:


df

