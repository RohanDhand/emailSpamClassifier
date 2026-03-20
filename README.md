# emailSpamClassifier

A machine learning project that classifies SMS/email messages as **spam** or **ham (not spam)** using Natural Language Processing and Naive Bayes algorithms.

---

##  Dataset

- **Source**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size**: 5,572 messages → 5,169 after removing 403 duplicates
- **Class Distribution**: 87.37% Ham | 12.63% Spam (imbalanced)

---

## Project Pipeline

```
Raw Data → Data Cleaning → EDA → Text Preprocessing → Model Building → Evaluation
```

### 1. Data Cleaning
- Dropped irrelevant unnamed columns
- Renamed columns to `target` and `text`
- Label encoded target: `ham = 0`, `spam = 1`
- Removed 403 duplicate entries

### 2. Exploratory Data Analysis (EDA)
- Class distribution via pie chart
- Added feature columns: `chars`, `words`, `sentences`
- Compared spam vs ham across all three features
- Correlation heatmap and pairplot analysis

**Key Finding**: Spam messages are significantly longer — avg ~138 chars vs ~70 chars for ham.

### 3. Text Preprocessing (`transform_text`)
- Lowercasing
- Tokenization (`nltk.word_tokenize`)
- Removal of special characters (non-alphanumeric)
- Stop word and punctuation removal
- Porter Stemming

### 4. Feature Extraction
- **Bag of Words** using `CountVectorizer`
- Vocabulary size: 5,091 unique tokens

### 5. Model Building
Three Naive Bayes variants evaluated:

| Model | Accuracy | Precision | Notes |
|---|---|---|---|
| GaussianNB | 88.0% | 53.1% | High false positives |
| MultinomialNB | 96.4% | 83.4% | Good balance |
| **BernoulliNB** | **97.0%** | **97.3%** | ✅ Best model |

> **BernoulliNB chosen** — highest precision means almost no legitimate mail is wrongly flagged as spam.

---

## Project Structure

```
email-spam-classifier/
│
├── emailSpam.ipynb        # Main notebook
├── spam_dataset/
│   └── spam.csv           # Raw dataset
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 4. Run the notebook
Open `emailSpam.ipynb` in Jupyter or Google Colab and run all cells.

> **Note**: If using Colab, update the dataset path in `pd.read_csv(...)` to match your Google Drive mount path or directly use the kaggle source given.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
wordcloud
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Explainability (SHAP)

SHAP values are used to interpret model predictions — identifying which words most strongly push a message toward spam or ham.

```python
import shap
explainer = shap.LinearExplainer(bnb, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=cv.get_feature_names_out())
```

Top spam-driving words identified: `free`, `call`, `txt`, `claim`, `win`, `prize`, `urgent`

---

## Results

**Best Model: BernoulliNB**
```
Accuracy  : 97.0%
Precision : 97.3%

Confusion Matrix:
[[893   3]
 [ 28 110]]
```
- Only **3** ham messages incorrectly flagged as spam
- **110** spam messages correctly caught

---

## Future Improvements

- [ ] Try TF-IDF vectorization instead of Bag of Words
- [ ] Experiment with ensemble models (Random Forest, XGBoost)
- [ ] Handle class imbalance with SMOTE or class weighting
- [ ] Deploy as a web app using Streamlit or FastAPI
- [ ] Add a Transformer-based model (DistilBERT) for comparison

---

