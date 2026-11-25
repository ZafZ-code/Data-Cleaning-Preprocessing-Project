# Predicting Subscriptions from YouTube Viewing Behavior

Machine learning project on a synthetic YouTube recommendation dataset. The goal is to clean the data, handle a strong class imbalance, compare several models and balancing strategies, and test whether viewing behavior predicts `subscribed_after`.

**Main result: no model and no feature separates the classes (ROC-AUC ≈ 0.50).** The project documents how this was checked, instead of reporting an inflated score.

## Dataset

- Source: [YouTube Recommendation Data for Cleaning and ML (Kaggle, iitanshravan)](https://www.kaggle.com/datasets/iitanshravan/youtube-recommendation-data-for-cleaning-and-ml)
- Size: about 800k rows in the training set, 26 features after preprocessing, 99,990 users and 50,000 videos
- Target: `subscribed_after`, about 5% positive (strong imbalance)

To reproduce, download the CSV from Kaggle and put it in a `Data/` folder.

## Pipeline

### 1. Cleaning

- **Timestamps:** parsed with an explicit format into a `datetime64` column, which makes them easy to manipulate, then split into a date column and a time column.
- **Missing and invalid values:** handled column by column. Examples of invalid values: negative `video_duration` or `watch_time`, and infinite (`-inf` / `inf`) `watch_percent`.
- **`liked` column:** it contained the modalities `1`, `0`, `yes`, `no`, `2` and `NaN`.
  - `yes` and `no` were directly converted to `1` and `0`.
  - `2` and `NaN` were converted to `0` / `1` with explicit business rules based on `watch_percent`, `commented`, `recommended` and `category`.
  - Rows matching no rule were dropped.

### 2. Feature engineering

- One-hot encoding of `device` and `category` (prefixes `dev_` and `cat_`).
- Time features from the timestamp: hour, day of week, month, year, and `is_weekend`.
- Cyclical sin/cos encodings of the hour, day of week and month.

### 3. Imbalance handling (train set only)

Four strategies were defined:

- `original`: no resampling
- `OverSampling`: ADASYN
- `UnderSampling`: NearMiss
- `combine`: ADASYN followed by NearMiss. It was **not run**, because it needs too many resources on about 800k training rows.

### 4. Models

RandomForest, ExtraTrees, XGBoost, LightGBM.

### 5. Evaluation

On the untouched test set: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC and precision-recall curves, and classification reports.

## Results

| Check | Finding |
|---|---|
| ROC-AUC, 4 models × 3 strategies | 0.499 to 0.504 (random-guess level) |
| `original` strategy | Accuracy 0.95, Recall about 0, F1 about 0: the model predicts only the majority class |
| Best F1 (about 0.095) | Equals the score of predicting "positive" for every row, so it is a baseline, not a learned pattern |
| Over-sampled models | Train accuracy far above test accuracy (overfitting to synthetic rows) |
| Single-feature ROC-AUC | The strongest feature is only 0.0028 away from 0.5 |
| Subscription rate by `liked` (0 vs 1) | 0.04998 vs 0.05005 (identical) |
| Subscription rate by `watch_percent` bins | About 0.050 in every bin |
| `watch_percent` > 1 (1,075 rows) | 0.0419, about 1.2 standard deviations below the base rate: consistent with noise |

## Conclusion

With these features, `subscribed_after` is not predictable. Every group has the same subscription rate as the overall base rate (5%).

**Why:** the exploratory analysis showed that all the features have the same proportions and distributions across groups. The data is synthetic and does not represent real user behavior, so there is no relationship for a model to learn.

GridSearchCV was not pursued: tuning a model whose ROC-AUC is 0.50 only optimizes noise.

## Lessons

- Accuracy is misleading at 5% positives: 0.95 is reached by predicting one class.
- F1 can look "best" for a model that labels every row positive. Always compare with ROC-AUC, PR-AUC and a baseline.
- Resampling must happen inside cross-validation folds (imblearn `Pipeline`), never before the split.
- Before tuning, check that any signal exists: per-feature AUC, group rates, and a noise test on small groups.

## Possible next steps

- Per-user and per-video features (rows per user, average watch percent, like rate), tested with out-of-fold target encoding.
- A dataset with a real relationship between behavior and subscription.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, matplotlib, seaborn. Run in Google Colab.

## Repository structure

```
├── Data/              # not tracked (download from Kaggle)
├── notebooks.ipynb
└── README.md
```