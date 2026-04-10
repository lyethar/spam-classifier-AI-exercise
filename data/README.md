# Dataset — SMS Spam Collection

## Download Instructions

The UCI SMS Spam Collection dataset is not included in this repository due to licensing. Download it as follows:

### Option A — Direct Download (Terminal)

```bash
cd data/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
mv SMSSpamCollection spam.csv
```

### Option B — Manual Download

1. Visit: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Click **Download**
3. Extract the zip
4. Rename `SMSSpamCollection` → `spam.csv`
5. Place it in this `data/` folder

## Dataset Details

| Property       | Value                           |
|----------------|---------------------------------|
| Source         | UCI Machine Learning Repository |
| Format         | Tab-separated (.txt)            |
| Total messages | 5,574                           |
| Spam messages  | 747 (13.4%)                     |
| Ham messages   | 4,827 (86.6%)                   |
| Language       | English                         |
| Collected      | 2011–2012                       |

## Class Imbalance Note

The dataset is imbalanced — only ~13% of messages are spam. This is realistic (most real SMS messages are legitimate) but means a naive model that always predicts "ham" would achieve 86.6% accuracy. Always evaluate with precision, recall, and F1-score, not just accuracy.
