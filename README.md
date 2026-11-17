# Anomaly Detection in Credit Card Transactions

Unsupervised fraud detection on the ULB / Kaggle credit-card dataset. We never train on the `Class` label. We use the label only after the models are fit, to measure precision, recall, and F1 on rare frauds.

The published file has **284,807** transactions. Features are the anonymized principal components `V1`–`V28`, plus `Time` and `Amount`. About **0.172%** of rows are fraud.

## What the notebook covers

| Assignment step | Where |
| --- | --- |
| Explore `Amount`, `Time`, and imbalance | Sections 2–3 |
| Scale features and hold out a test set | Section 4 |
| PCA and t-SNE pictures | Section 5 |
| Isolation Forest, One-Class SVM, autoencoder | Section 6 |
| Precision, recall, F1 on hidden labels | Section 7 |
| What got flagged, and which patterns show up | Section 8 |
| Conclusions | Section 9 |

## Method in one paragraph

Each model learns what a *typical* transaction looks like. It never sees `Class`. We rank transactions by how unusual they are. Before looking at test labels, we fix the review budget at **0.2%** of transactions (about two reviews per thousand). We freeze the cutoff that hits that budget on a calibration slice of the training data, then apply it to the test set. We compute precision, recall, and F1 only then. We also report average precision, because accuracy is misleading when almost every row is legitimate.

## Project layout

```
credit_card_anomaly_detection.ipynb
requirements.txt
README.md
data/creditcard.csv                   # downloaded from Kaggle; gitignored
figures/                              # plots written by the notebook
results/metrics.csv                   # written by the notebook
results/summary.txt
```

## Setup (Windows)

1. We create a Kaggle account, open the dataset page, and download `creditcard.csv`.
2. We put it in the `data` folder next to the notebook.
3. In PowerShell, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook credit_card_anomaly_detection.ipynb
```

We can also open the notebook in Cursor and run all cells. The working directory is this project folder.

The first code cell checks that `data/creditcard.csv` exists and stops with a message if it does not.

## Runtime

On a normal laptop the full run is usually several minutes. The One-Class SVM cell is the slow one: we fit the RBF model on 5,000 rows (the classic solver does not scale to 280,000 rows) and then score the calibration and test rows in batches. Isolation Forest and the autoencoder train on about 90% of the training rows.

Knobs at the top of the notebook:

- `ALERT_RATE` — review budget. We leave this at `0.002` for the submission so the budget is not tuned on the labels.
- `RUN_TSNE` — we set this to `False` when we want to skip the t-SNE plot.
- `AE_EPOCHS` — we lower this if the autoencoder is taking too long.
- `IF_N_JOBS` — we set this to `1` if Windows reports a parallel-processing error.
- `RBF_TRAIN_ROWS` — we lower this if the SVM runs out of memory. Scoring gets cheaper when the model is fit on fewer rows.

## How labels are used

| Step | Sees `Class`? |
| --- | --- |
| Cleaning, Amount and Time plots, correlations | No, except the class-balance chart, which is display only |
| `MinMaxScaler`, train / calibration / test split | No |
| Isolation Forest, One-Class SVM, autoencoder | No |
| Choice of the 0.2% alert rate | No (fixed before scores are inspected) |
| Precision, recall, F1, confusion matrices, fraud profiles | Yes |

The split is random, not stratified, so the split itself does not use the label.

## Models

- **Isolation Forest** isolates unusual points with random trees. A point that is separated in few splits gets a high score. We fit it on the large training slice.
- **One-Class SVM** (RBF kernel) draws a boundary around the dense region. We fit it on 5,000 training rows because a full RBF fit is not practical at this size. `nu=0.05` is the optimizer setting (how many training points may fall outside the boundary). The alert rule is still the shared 0.2% cutoff, not the SVM’s default boundary.
- **Autoencoder** (`MLPRegressor` trained to reconstruct its input through a bottleneck of 8 units). The score is reconstruction error. Most training rows are legitimate, so the network spends its capacity on normal patterns. Rows it cannot rebuild rise to the top of the list.

We compare all three on the **same test rows** at the **same alert rate**. The SVM saw fewer training rows; that limit is stated in the notebook.

## Metrics that matter

- **Recall** — of the real frauds in the test set, how many were flagged.
- **Precision** — of the flags, how many were real frauds.
- **F1** — the balance of those two at the 0.2% budget.
- **Average precision** — ranking quality across budgets. A model that ranks at random scores about the fraud rate (near 0.002).

ROC-AUC is printed and plotted with a warning. It can look strong even when a reviewer would still wade through many false alarms.

## Submission

1. Run all cells from top to bottom.
2. Confirm the metric table, confusion matrices, and `results/summary.txt`.
3. Save the notebook with outputs.
4. Create a public GitHub repository (name: `anomaly-detection-credit-card`).
5. Push this project. Leave `creditcard.csv` out of the repo.
6. Submit the repository link through the form in the course post.

Saved outputs stay in the notebook so the work can be read without a re-run. An un-run notebook is incomplete.

## Dataset citation

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium on Computational Intelligence and Data Mining, 2015.

Kaggle mirror: [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## What this shows

- The models detect unusual transactions. Fraud is one reason a transaction is unusual, not the definition they learned.
- The alert rate was chosen first. Moving it after seeing F1 would use the labels to tune the system.
- Precision stays modest because rare legitimate purchases also look unusual.
- Isolation Forest scales to the full training slice. Classic RBF One-Class SVM does not, which is why we fit it on a sample.
- `V1`–`V28` are already principal components. A 2D scatter will not show the whole story, and those components have no public meaning (merchant, country, and so on are not in the file).
