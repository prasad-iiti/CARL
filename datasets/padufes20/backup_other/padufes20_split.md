```python
import os
import pandas as pd
import numpy as np
```


```python
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
RANDOM_SEED = 123
```


```python
rng = np.random.default_rng(RANDOM_SEED)
```


```python
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6
```


```python
df = pd.read_csv("padufes20_metadata_official.csv")
```


```python
df.columns
```




    Index(['patient_id', 'lesion_id', 'smoke', 'drink', 'background_father',
           'background_mother', 'age', 'pesticide', 'gender',
           'skin_cancer_history', 'cancer_history', 'has_piped_water',
           'has_sewage_system', 'fitspatrick', 'region', 'diameter_1',
           'diameter_2', 'diagnostic', 'itch', 'grew', 'hurt', 'changed', 'bleed',
           'elevation', 'img_id', 'biopsed'],
          dtype='str')




```python
keep_cols = [
    "img_id",
    "diagnostic",
    "patient_id",
    "age",
    "gender",
    "region",
]
df = df[keep_cols]
```


```python
df["diagnostic"].value_counts(dropna="FALSE")
```




    diagnostic
    BCC    845
    ACK    730
    NEV    244
    SEK    235
    SCC    192
    MEL     52
    Name: count, dtype: int64




```python
df["dataset"] = "padufes20"
```


```python
df.rename(columns={
    "img_id": "image_name",
    "diagnostic": "label",
    "gender": "sex",
    "region": "location",
}, inplace=True)
```


```python
df["sex"] = df["sex"].fillna("unknown")
```


```python
df.isna().sum()
```




    image_name    0
    label         0
    patient_id    0
    age           0
    sex           0
    location      0
    dataset       0
    dtype: int64




```python
assert df.isnull().sum().sum() == 0
```


```python
assert len(df) == df["image_name"].nunique()
```


```python
missing = []
for f in df["image_name"]:
    if not os.path.exists(os.path.join("/home/prasad/datasets/padufes20/images/", f)):
        missing.append(f)

print(len(missing))
```

    0



```python
sorted(df["label"].unique().tolist())
```




    ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']




```python
mapping = {
    "ACK": "AKIEC",
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NEV",
    "SCC": "SCC",
    "SEK": "SEK",
}
```


```python
assert set(df["label"].unique()) == set(mapping.keys())
```


```python
df["label"] = df["label"].map(mapping)
```


```python
assert df["label"].isna().sum() == 0
```


```python
assert sorted(df["label"].unique()) == [
    "AKIEC", "BCC", "MEL", "NEV", "SCC", "SEK"]
```


```python
df["label"].value_counts()
```




    label
    BCC      845
    AKIEC    730
    NEV      244
    SEK      235
    SCC      192
    MEL       52
    Name: count, dtype: int64




```python
print(df["label"].value_counts().sum())
```

    2298



```python
df["binary_label"]=(df["label"].isin(
    ["AKIEC", "MEL", "BCC", "SCC"])).astype(int)
```


```python
assert set(df["binary_label"].unique()) == {0,1}
```


```python
print(df["binary_label"].value_counts())
```

    binary_label
    1    1819
    0     479
    Name: count, dtype: int64



```python
print(df["binary_label"].value_counts().sum())
```

    2298



```python
print(pd.crosstab(df["binary_label"], df["label"]))
```

    label         AKIEC  BCC  MEL  NEV  SCC  SEK
    binary_label                                
    0                 0    0    0  244    0  235
    1               730  845   52    0  192    0



```python
df["sex"].value_counts()
```




    sex
    unknown    804
    FEMALE     753
    MALE       741
    Name: count, dtype: int64




```python
print(df["sex"].value_counts().sum())
```

    2298



```python
print(pd.crosstab(df["sex"], df["label"]))
```

    label    AKIEC  BCC  MEL  NEV  SCC  SEK
    sex                                    
    FEMALE     153  406   28   49   90   27
    MALE       130  439   24   26  102   20
    unknown    447    0    0  169    0  188



```python
df["age"].value_counts()
```




    age
    55    73
    73    73
    58    70
    75    66
    62    66
          ..
    8      2
    24     2
    12     2
    6      2
    18     1
    Name: count, Length: 84, dtype: int64




```python
df["age"].value_counts().sum()
```




    np.int64(2298)




```python
df["age"].isna().sum()
```




    np.int64(0)




```python
print(pd.crosstab(df["age"], df["label"]))
```

    label  AKIEC  BCC  MEL  NEV  SCC  SEK
    age                                  
    6          0    0    0    2    0    0
    8          0    0    0    2    0    0
    9          0    0    0    3    0    0
    10         0    0    0    5    0    0
    12         0    0    0    2    0    0
    ..       ...  ...  ...  ...  ...  ...
    89         2    4    0    0    0    1
    90         2    3    2    0    4    3
    91         0    5    0    0    2    0
    92         1    0    0    0    3    0
    94         0    2    0    0    1    0
    
    [84 rows x 6 columns]



```python
df["location"].value_counts()
```




    location
    FACE       570
    FOREARM    392
    CHEST      280
    BACK       248
    ARM        192
    NOSE       158
    HAND       126
    NECK        93
    THIGH       73
    EAR         73
    ABDOMEN     36
    LIP         23
    SCALP       18
    FOOT        16
    Name: count, dtype: int64




```python
df["location"].value_counts().sum()
```




    np.int64(2298)




```python
df["location"].isna().sum()
```




    np.int64(0)




```python
print(pd.crosstab(df["location"], df["label"]))
```

    label     AKIEC  BCC  MEL  NEV  SCC  SEK
    location                                
    ABDOMEN       3    2    2   19    2    8
    ARM          91   44    7   10   21   19
    BACK         16   91   14   86   17   24
    CHEST        60  135    5   29   20   31
    EAR          11   36    2    3   13    8
    FACE        139  260    5   38   29   99
    FOOT          4    0    0    6    6    0
    FOREARM     261   67    5   10   28   21
    HAND         78   19    2    6   19    2
    LIP           4   10    0    3    6    0
    NECK         18   48    3    8   10    6
    NOSE         23  123    0    0    9    3
    SCALP         4    0    0    9    1    4
    THIGH        18   10    7   17   11   10


# Split


```python
len(df["patient_id"])
```




    2298




```python
patient_ids = df["patient_id"].dropna().unique().to_numpy()
```


```python
rng.shuffle(patient_ids)
```


```python
df["patient_id"].isna().sum()
```




    np.int64(0)




```python
n_patients = len(patient_ids)
n_train = int(TRAIN_RATIO * n_patients)
n_val = int(VAL_RATIO * n_patients)
```


```python
train_patients = patient_ids[:n_train]
val_patients = patient_ids[n_train:n_train + n_val]
test_patients = patient_ids[n_train + n_val:]
```


```python
train_ids = set(train_patients)
val_ids   = set(val_patients)
test_ids  = set(test_patients)

print("Train ∩ Val :", train_ids & val_ids)
print("Train ∩ Test:", train_ids & test_ids)
print("Val ∩ Test  :", val_ids & test_ids)
```

    Train ∩ Val : set()
    Train ∩ Test: set()
    Val ∩ Test  : set()



```python
df["split"] = "unassigned"
df.loc[df["patient_id"].isin(train_patients), "split"] = "train"
df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
df.loc[df["patient_id"].isin(test_patients), "split"] = "test"
```


```python
assert (df["split"] != "unassigned").all()
```


```python
print("Patients per split:")
print(df.groupby("split")["patient_id"].nunique())
```

    Patients per split:
    split
    test     275
    train    961
    val      137
    Name: patient_id, dtype: int64



```python
print("\nImages per split:")
print(df["split"].value_counts())
```

    
    Images per split:
    split
    train    1620
    test      457
    val       221
    Name: count, dtype: int64



```python
print("\nClass distribution per split:")
print(pd.crosstab(df["split"], df["label"]))
```

    
    Class distribution per split:
    label  AKIEC  BCC  MEL  NEV  SCC  SEK
    split                                
    test     139  162   18   46   43   49
    train    520  611   30  166  133  160
    val       71   72    4   32   16   26



```python
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
```

    label  AKIEC  BCC  MEL  NEV  SCC  SEK
    split                                
    test     139  162   18   46   43   49
    train    520  611   30  166  133  160
    val       71   72    4   32   16   26



```python
pd.crosstab(df["age"], df["split"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>split</th>
      <th>test</th>
      <th>train</th>
      <th>val</th>
    </tr>
    <tr>
      <th>age</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0</td>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 3 columns</p>
</div>




```python
pd.crosstab(df["sex"], df["split"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>split</th>
      <th>test</th>
      <th>train</th>
      <th>val</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FEMALE</th>
      <td>159</td>
      <td>536</td>
      <td>58</td>
    </tr>
    <tr>
      <th>MALE</th>
      <td>142</td>
      <td>524</td>
      <td>75</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>156</td>
      <td>560</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.crosstab(df["location"], df["split"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>split</th>
      <th>test</th>
      <th>train</th>
      <th>val</th>
    </tr>
    <tr>
      <th>location</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ABDOMEN</th>
      <td>8</td>
      <td>23</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ARM</th>
      <td>51</td>
      <td>126</td>
      <td>15</td>
    </tr>
    <tr>
      <th>BACK</th>
      <td>53</td>
      <td>170</td>
      <td>25</td>
    </tr>
    <tr>
      <th>CHEST</th>
      <td>41</td>
      <td>205</td>
      <td>34</td>
    </tr>
    <tr>
      <th>EAR</th>
      <td>14</td>
      <td>55</td>
      <td>4</td>
    </tr>
    <tr>
      <th>FACE</th>
      <td>126</td>
      <td>398</td>
      <td>46</td>
    </tr>
    <tr>
      <th>FOOT</th>
      <td>0</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>FOREARM</th>
      <td>78</td>
      <td>275</td>
      <td>39</td>
    </tr>
    <tr>
      <th>HAND</th>
      <td>24</td>
      <td>89</td>
      <td>13</td>
    </tr>
    <tr>
      <th>LIP</th>
      <td>6</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NECK</th>
      <td>11</td>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>NOSE</th>
      <td>22</td>
      <td>121</td>
      <td>15</td>
    </tr>
    <tr>
      <th>SCALP</th>
      <td>3</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>THIGH</th>
      <td>20</td>
      <td>46</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(["split", "sex", "label"]).size()
```




    split  sex      label
    test   FEMALE   AKIEC     39
                    BCC       79
                    MEL       10
                    NEV        8
                    SCC       20
                    SEK        3
           MALE     AKIEC     21
                    BCC       83
                    MEL        8
                    NEV        1
                    SCC       23
                    SEK        6
           unknown  AKIEC     79
                    NEV       37
                    SEK       40
    train  FEMALE   AKIEC    105
                    BCC      295
                    MEL       18
                    NEV       32
                    SCC       62
                    SEK       24
           MALE     AKIEC     92
                    BCC      316
                    MEL       12
                    NEV       24
                    SCC       71
                    SEK        9
           unknown  AKIEC    323
                    NEV      110
                    SEK      127
    val    FEMALE   AKIEC      9
                    BCC       32
                    NEV        9
                    SCC        8
           MALE     AKIEC     17
                    BCC       40
                    MEL        4
                    NEV        1
                    SCC        8
                    SEK        5
           unknown  AKIEC     45
                    NEV       22
                    SEK       21
    dtype: int64




```python
assert df["image_name"].is_unique
assert df["split"].isin(["train", "val", "test"]).all()
assert df["sex"].isin(["MALE", "FEMALE", "unknown"]).all()
assert df["label"].notna().all()
```


```python
leakage = (
    df.groupby("patient_id")["split"]
    .nunique()
    .gt(1)
    .sum()
)
print(f"Patients appearing in multiple splits: {leakage}")
```

    Patients appearing in multiple splits: 0



```python
df=df[["dataset", "split", "image_name", "label", "binary_label", "patient_id",
        "age", "sex", "location"]]
```


```python
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
```


```python
print("=" * 60)
print("PADUFES20 metadata preprocessing completed")
print(f"Images   : {len(df)}")
print(f"Patients : {df['patient_id'].nunique()}")
print(f"Classes  : {df['label'].nunique()}")
print(df["split"].value_counts())
print("=" * 60)
```

    ============================================================
    PADUFES20 metadata preprocessing completed
    Images   : 2298
    Patients : 1373
    Classes  : 6
    split
    train    1620
    test      457
    val       221
    Name: count, dtype: int64
    ============================================================



```python
print(df.isnull().sum())
```

    dataset         0
    split           0
    image_name      0
    label           0
    binary_label    0
    patient_id      0
    age             0
    sex             0
    location        0
    dtype: int64



```python
assert df["image_name"].is_unique
assert df["patient_id"].notna().all()
assert df["label"].notna().all()
assert df["binary_label"].isin([0, 1]).all()
assert df["split"].isin(["train", "val", "test"]).all()
```


```python
assert df.isnull().sum().sum() == 0
```


```python
df.to_csv("padufes20_metadata_clean.csv", index=False, encoding="utf-8")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>split</th>
      <th>image_name</th>
      <th>label</th>
      <th>binary_label</th>
      <th>patient_id</th>
      <th>age</th>
      <th>sex</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>padufes20</td>
      <td>train</td>
      <td>PAT_1516_1765_530.png</td>
      <td>NEV</td>
      <td>0</td>
      <td>PAT_1516</td>
      <td>8</td>
      <td>unknown</td>
      <td>ARM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>padufes20</td>
      <td>train</td>
      <td>PAT_46_881_939.png</td>
      <td>BCC</td>
      <td>1</td>
      <td>PAT_46</td>
      <td>55</td>
      <td>FEMALE</td>
      <td>NECK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>padufes20</td>
      <td>val</td>
      <td>PAT_1545_1867_547.png</td>
      <td>AKIEC</td>
      <td>1</td>
      <td>PAT_1545</td>
      <td>77</td>
      <td>unknown</td>
      <td>FACE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>padufes20</td>
      <td>test</td>
      <td>PAT_1989_4061_934.png</td>
      <td>AKIEC</td>
      <td>1</td>
      <td>PAT_1989</td>
      <td>75</td>
      <td>unknown</td>
      <td>HAND</td>
    </tr>
    <tr>
      <th>4</th>
      <td>padufes20</td>
      <td>train</td>
      <td>PAT_684_1302_588.png</td>
      <td>BCC</td>
      <td>1</td>
      <td>PAT_684</td>
      <td>79</td>
      <td>MALE</td>
      <td>FOREARM</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2293</th>
      <td>padufes20</td>
      <td>train</td>
      <td>PAT_1708_3156_175.png</td>
      <td>AKIEC</td>
      <td>1</td>
      <td>PAT_1708</td>
      <td>73</td>
      <td>unknown</td>
      <td>HAND</td>
    </tr>
    <tr>
      <th>2294</th>
      <td>padufes20</td>
      <td>train</td>
      <td>PAT_46_880_140.png</td>
      <td>BCC</td>
      <td>1</td>
      <td>PAT_46</td>
      <td>55</td>
      <td>FEMALE</td>
      <td>CHEST</td>
    </tr>
    <tr>
      <th>2295</th>
      <td>padufes20</td>
      <td>val</td>
      <td>PAT_1343_1217_404.png</td>
      <td>SEK</td>
      <td>0</td>
      <td>PAT_1343</td>
      <td>74</td>
      <td>unknown</td>
      <td>FOREARM</td>
    </tr>
    <tr>
      <th>2296</th>
      <td>padufes20</td>
      <td>test</td>
      <td>PAT_326_690_823.png</td>
      <td>BCC</td>
      <td>1</td>
      <td>PAT_326</td>
      <td>58</td>
      <td>FEMALE</td>
      <td>FACE</td>
    </tr>
    <tr>
      <th>2297</th>
      <td>padufes20</td>
      <td>train</td>
      <td>PAT_1714_3189_989.png</td>
      <td>SEK</td>
      <td>0</td>
      <td>PAT_1714</td>
      <td>41</td>
      <td>unknown</td>
      <td>THIGH</td>
    </tr>
  </tbody>
</table>
<p>2298 rows × 9 columns</p>
</div>


