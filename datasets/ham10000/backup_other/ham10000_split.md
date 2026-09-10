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
df = pd.read_csv("HAM10000_metadata_official.csv")
```


```python
df.drop(columns=["dx_type", "dataset"], inplace=True)
```


```python
df["dataset"] = "ham10000"
```


```python
df.rename(columns={
    "image_id": "image_name",
    "dx": "label",
    "lesion_id": "patient_id",
    "localization": "location",
}, inplace=True)
```


```python
df["age"] = df["age"].fillna("unknown")
```


```python
df.isna().sum()
```




    patient_id    0
    image_name    0
    label         0
    age           0
    sex           0
    location      0
    dataset       0
    dtype: int64




```python
assert df.isnull().sum().sum() == 0
```


```python
df["image_name"] = df["image_name"].apply(
    lambda x: x if str(x).endswith(".jpg") else str(x) + ".jpg")
```


```python
assert len(df) == df["image_name"].nunique()
```


```python
missing = []
for f in df["image_name"]:
    if not os.path.exists(os.path.join("/home/prasad/datasets/ham10000/images/", f)):
        missing.append(f)

print(len(missing))
```

    0



```python
mapping = {
    "akiec": "AKIEC",
    "bcc"  : "BCC",
    "bkl"  : "BKL",
    "df"   : "DF",
    "mel"  : "MEL",
    "nv"   : "NEV",
    "vasc" : "VASC",
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
    "AKIEC", "BCC", "BKL", "DF", "MEL", "NEV", "VASC"]
```


```python
df["label"].value_counts()
```




    label
    NEV      6705
    MEL      1113
    BKL      1099
    BCC       514
    AKIEC     327
    VASC      142
    DF        115
    Name: count, dtype: int64




```python
print(df["label"].value_counts().sum())
```

    10015



```python
assert df["label"].isna().sum() == 0
```


```python
df["binary_label"]=(df["label"].isin(
    ["MEL", "BCC", "AKIEC"])).astype(int)
```


```python
assert set(df["binary_label"].unique()) == {0,1}
```


```python
print(df["binary_label"].value_counts())
```

    binary_label
    0    8061
    1    1954
    Name: count, dtype: int64



```python
print(df["binary_label"].value_counts().sum())
```

    10015



```python
print(pd.crosstab(df["binary_label"], df["label"]))
```

    label         AKIEC  BCC   BKL   DF   MEL   NEV  VASC
    binary_label                                         
    0                 0    0  1099  115     0  6705   142
    1               327  514     0    0  1113     0     0



```python
df["sex"].value_counts()
```




    sex
    male       5406
    female     4552
    unknown      57
    Name: count, dtype: int64




```python
print(df["sex"].value_counts().sum())
```

    10015



```python
print(pd.crosstab(df["sex"], df["label"]))
```

    label    AKIEC  BCC  BKL  DF  MEL   NEV  VASC
    sex                                          
    female     106  197  463  52  424  3237    73
    male       221  317  626  63  689  3421    69
    unknown      0    0   10   0    0    47     0



```python
df["age"].value_counts()
```




    age
    45.0       1299
    50.0       1187
    55.0       1009
    40.0        985
    60.0        803
    70.0        756
    35.0        753
    65.0        731
    75.0        618
    30.0        464
    80.0        404
    85.0        290
    25.0        247
    20.0        169
    5.0          86
    15.0         77
    unknown      57
    10.0         41
    0.0          39
    Name: count, dtype: int64




```python
df["age"].value_counts().sum()
```




    np.int64(10015)




```python
df["age"].isna().sum()
```




    np.int64(0)




```python
print(pd.crosstab(df["age"], df["label"]))
```

    label    AKIEC  BCC  BKL  DF  MEL   NEV  VASC
    age                                          
    0.0          0    0    5   0    0    30     4
    5.0          0    0    1   0    1    81     3
    10.0         0    0    0   0    0    39     2
    15.0         0    0    0   0    0    73     4
    20.0         0    1    0   0    6   158     4
    25.0         0    3    0   2   16   221     5
    30.0         1    4    6   4   34   410     5
    35.0         0    5   24  12   36   668     8
    40.0         9   23   46   9   49   846     3
    45.0        10   26   59  14   74  1100    16
    50.0        19   27   87  18   96   928    12
    55.0        27   25   95  13  142   686    21
    60.0        58   35  131   9  106   454    10
    65.0        38   79  108  18  133   351     4
    70.0        56   85  183   4  166   248    14
    75.0        47   76  153   9   91   231    11
    80.0        37   73   98   3   85    97    11
    85.0        25   52   93   0   76    39     5
    unknown      0    0   10   0    2    45     0



```python
df["location"].value_counts()
```




    location
    back               2192
    lower extremity    2077
    trunk              1404
    upper extremity    1118
    abdomen            1022
    face                745
    chest               407
    foot                319
    unknown             234
    neck                168
    scalp               128
    hand                 90
    ear                  56
    genital              48
    acral                 7
    Name: count, dtype: int64




```python
df["location"].value_counts().sum()
```




    np.int64(10015)




```python
df["location"].isna().sum()
```




    np.int64(0)




```python
print(pd.crosstab(df["location"], df["label"]))
```

    label            AKIEC  BCC  BKL  DF  MEL   NEV  VASC
    location                                             
    abdomen              5   18   42   4   66   860    27
    acral                0    0    0   0    0     7     0
    back                29  186  202   2  324  1427    22
    chest               12   47   74   0   68   200     6
    ear                  3    0    6   0   17    30     0
    face               113  101  319   0  104   100     8
    foot                 0    4    1   2   28   284     0
    genital              0    0    2   1    0    45     0
    hand                13    2   14   0    1    55     5
    lower extremity     65   58  174  82  192  1479    27
    neck                10   14   31   0   29    81     3
    scalp               14   19   34   0   14    45     2
    trunk                1   11   73   0   47  1241    31
    unknown              0    5   23   0   10   196     0
    upper extremity     62   49  104  24  213   655    11


# Split


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
    test     1494
    train    5229
    val       747
    Name: patient_id, dtype: int64



```python
print("\nImages per split:")
print(df["split"].value_counts())
```

    
    Images per split:
    split
    train    7034
    test     1982
    val       999
    Name: count, dtype: int64



```python
print("\nClass distribution per split:")
print(pd.crosstab(df["split"], df["label"]))
```

    
    Class distribution per split:
    label  AKIEC  BCC  BKL  DF  MEL   NEV  VASC
    split                                      
    test      72  109  194  24  220  1342    21
    train    218  357  783  82  788  4699   107
    val       37   48  122   9  105   664    14



```python
print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
```

    label  AKIEC  BCC  BKL  DF  MEL   NEV  VASC
    split                                      
    test      72  109  194  24  220  1342    21
    train    218  357  783  82  788  4699   107
    val       37   48  122   9  105   664    14



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
      <th>0.0</th>
      <td>8</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>17</td>
      <td>64</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>13</td>
      <td>24</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15.0</th>
      <td>11</td>
      <td>56</td>
      <td>10</td>
    </tr>
    <tr>
      <th>20.0</th>
      <td>29</td>
      <td>115</td>
      <td>25</td>
    </tr>
    <tr>
      <th>25.0</th>
      <td>52</td>
      <td>160</td>
      <td>35</td>
    </tr>
    <tr>
      <th>30.0</th>
      <td>90</td>
      <td>325</td>
      <td>49</td>
    </tr>
    <tr>
      <th>35.0</th>
      <td>139</td>
      <td>529</td>
      <td>85</td>
    </tr>
    <tr>
      <th>40.0</th>
      <td>190</td>
      <td>704</td>
      <td>91</td>
    </tr>
    <tr>
      <th>45.0</th>
      <td>259</td>
      <td>928</td>
      <td>112</td>
    </tr>
    <tr>
      <th>50.0</th>
      <td>244</td>
      <td>834</td>
      <td>109</td>
    </tr>
    <tr>
      <th>55.0</th>
      <td>182</td>
      <td>701</td>
      <td>126</td>
    </tr>
    <tr>
      <th>60.0</th>
      <td>167</td>
      <td>567</td>
      <td>69</td>
    </tr>
    <tr>
      <th>65.0</th>
      <td>147</td>
      <td>530</td>
      <td>54</td>
    </tr>
    <tr>
      <th>70.0</th>
      <td>161</td>
      <td>506</td>
      <td>89</td>
    </tr>
    <tr>
      <th>75.0</th>
      <td>131</td>
      <td>437</td>
      <td>50</td>
    </tr>
    <tr>
      <th>80.0</th>
      <td>65</td>
      <td>288</td>
      <td>51</td>
    </tr>
    <tr>
      <th>85.0</th>
      <td>64</td>
      <td>198</td>
      <td>28</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>13</td>
      <td>37</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
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
      <th>female</th>
      <td>906</td>
      <td>3169</td>
      <td>477</td>
    </tr>
    <tr>
      <th>male</th>
      <td>1066</td>
      <td>3825</td>
      <td>515</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>10</td>
      <td>40</td>
      <td>7</td>
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
      <th>abdomen</th>
      <td>216</td>
      <td>712</td>
      <td>94</td>
    </tr>
    <tr>
      <th>acral</th>
      <td>0</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>back</th>
      <td>410</td>
      <td>1571</td>
      <td>211</td>
    </tr>
    <tr>
      <th>chest</th>
      <td>98</td>
      <td>268</td>
      <td>41</td>
    </tr>
    <tr>
      <th>ear</th>
      <td>8</td>
      <td>41</td>
      <td>7</td>
    </tr>
    <tr>
      <th>face</th>
      <td>143</td>
      <td>530</td>
      <td>72</td>
    </tr>
    <tr>
      <th>foot</th>
      <td>78</td>
      <td>211</td>
      <td>30</td>
    </tr>
    <tr>
      <th>genital</th>
      <td>14</td>
      <td>32</td>
      <td>2</td>
    </tr>
    <tr>
      <th>hand</th>
      <td>22</td>
      <td>60</td>
      <td>8</td>
    </tr>
    <tr>
      <th>lower extremity</th>
      <td>418</td>
      <td>1442</td>
      <td>217</td>
    </tr>
    <tr>
      <th>neck</th>
      <td>32</td>
      <td>120</td>
      <td>16</td>
    </tr>
    <tr>
      <th>scalp</th>
      <td>24</td>
      <td>92</td>
      <td>12</td>
    </tr>
    <tr>
      <th>trunk</th>
      <td>273</td>
      <td>996</td>
      <td>135</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>48</td>
      <td>156</td>
      <td>30</td>
    </tr>
    <tr>
      <th>upper extremity</th>
      <td>198</td>
      <td>796</td>
      <td>124</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(["split", "sex", "label"]).size()
```




    split  sex      label
    test   female   AKIEC      29
                    BCC        50
                    BKL        84
                    DF          9
                    MEL        83
                    NEV       638
                    VASC       13
           male     AKIEC      43
                    BCC        59
                    BKL       107
                    DF         15
                    MEL       137
                    NEV       697
                    VASC        8
           unknown  BKL         3
                    NEV         7
    train  female   AKIEC      61
                    BCC       132
                    BKL       314
                    DF         40
                    MEL       305
                    NEV      2267
                    VASC       50
           male     AKIEC     157
                    BCC       225
                    BKL       464
                    DF         42
                    MEL       483
                    NEV      2397
                    VASC       57
           unknown  BKL         5
                    NEV        35
    val    female   AKIEC      16
                    BCC        15
                    BKL        65
                    DF          3
                    MEL        36
                    NEV       332
                    VASC       10
           male     AKIEC      21
                    BCC        33
                    BKL        55
                    DF          6
                    MEL        69
                    NEV       327
                    VASC        4
           unknown  BKL         2
                    NEV         5
    dtype: int64




```python
assert df["image_name"].is_unique
assert df["split"].isin(["train", "val", "test"]).all()
assert df["sex"].isin(["male", "female", "unknown"]).all()
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
print("HAM10000 metadata preprocessing completed")
print(f"Images   : {len(df)}")
print(f"Patients : {df['patient_id'].nunique()}")
print(f"Classes  : {df['label'].nunique()}")
print(df["split"].value_counts())
print("=" * 60)
```

    ============================================================
    HAM10000 metadata preprocessing completed
    Images   : 10015
    Patients : 7470
    Classes  : 7
    split
    train    7034
    test     1982
    val       999
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
df.to_csv("ham10000_metadata_clean.csv", index=False, encoding="utf-8")
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
      <td>ham10000</td>
      <td>train</td>
      <td>ISIC_0027419.jpg</td>
      <td>BKL</td>
      <td>0</td>
      <td>HAM_0000118</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham10000</td>
      <td>train</td>
      <td>ISIC_0025030.jpg</td>
      <td>BKL</td>
      <td>0</td>
      <td>HAM_0000118</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham10000</td>
      <td>val</td>
      <td>ISIC_0026769.jpg</td>
      <td>BKL</td>
      <td>0</td>
      <td>HAM_0002730</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham10000</td>
      <td>val</td>
      <td>ISIC_0025661.jpg</td>
      <td>BKL</td>
      <td>0</td>
      <td>HAM_0002730</td>
      <td>80.0</td>
      <td>male</td>
      <td>scalp</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham10000</td>
      <td>train</td>
      <td>ISIC_0031633.jpg</td>
      <td>BKL</td>
      <td>0</td>
      <td>HAM_0001466</td>
      <td>75.0</td>
      <td>male</td>
      <td>ear</td>
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
      <th>10010</th>
      <td>ham10000</td>
      <td>test</td>
      <td>ISIC_0033084.jpg</td>
      <td>AKIEC</td>
      <td>1</td>
      <td>HAM_0002867</td>
      <td>40.0</td>
      <td>male</td>
      <td>abdomen</td>
    </tr>
    <tr>
      <th>10011</th>
      <td>ham10000</td>
      <td>test</td>
      <td>ISIC_0033550.jpg</td>
      <td>AKIEC</td>
      <td>1</td>
      <td>HAM_0002867</td>
      <td>40.0</td>
      <td>male</td>
      <td>abdomen</td>
    </tr>
    <tr>
      <th>10012</th>
      <td>ham10000</td>
      <td>test</td>
      <td>ISIC_0033536.jpg</td>
      <td>AKIEC</td>
      <td>1</td>
      <td>HAM_0002867</td>
      <td>40.0</td>
      <td>male</td>
      <td>abdomen</td>
    </tr>
    <tr>
      <th>10013</th>
      <td>ham10000</td>
      <td>train</td>
      <td>ISIC_0032854.jpg</td>
      <td>AKIEC</td>
      <td>1</td>
      <td>HAM_0000239</td>
      <td>80.0</td>
      <td>male</td>
      <td>face</td>
    </tr>
    <tr>
      <th>10014</th>
      <td>ham10000</td>
      <td>train</td>
      <td>ISIC_0032258.jpg</td>
      <td>MEL</td>
      <td>1</td>
      <td>HAM_0003521</td>
      <td>70.0</td>
      <td>female</td>
      <td>back</td>
    </tr>
  </tbody>
</table>
<p>10015 rows × 9 columns</p>
</div>


