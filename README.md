# 📱 SMS Spam Classifier — AI Security Portfolio

> This project demonstrates how to build, train, and evaluate an SMS spam classification model using Python and scikit-learn — and critically, examines it through an **AI Red Teaming lens**.

---

## 🧠 What This Project Does

This classifier reads an SMS text message and predicts whether it is **spam** or **ham** (legitimate). It uses:

- **TF-IDF Vectorisation** to turn words into numbers a machine can understand
- **Multinomial Naive Bayes** as the classification algorithm
- The **UCI SMS Spam Collection Dataset** (5,574 real-world labelled messages)

Beyond the model itself, this repo includes red-team analysis: how an attacker could **evade**, **poison**, or **probe** this type of classifier — directly applicable to AI red teaming engagements.

---

## 📁 Repository Structure

```
spam-classifier-htb/
│
├── README.md                        # This file — setup guide + red team analysis
├── requirements.txt                 # Python dependencies
│
├── data/
│   └── README.md                    # Dataset download instructions
│
├── notebooks/
│   └── spam_classifier_walkthrough.ipynb   # Interactive Jupyter walkthrough
│
├── src/
│   ├── train.py                     # Train and save the model
│   ├── predict.py                   # Load model and classify new messages
│   └── evaluate.py                  # Detailed metrics and confusion matrix
│
└── tests/
    └── test_classifier.py           # Unit tests for the classifier
```

---

## ⚙️ Prerequisites

- Python 3.8+
- pip or conda
- ~200MB disk space

---

## 🚀 Step-by-Step Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/lyethar/spam-classifier-AI-exercise.git
cd spam-classifier-AI-exercise
```

### Step 2 — Create a Virtual Environment (Recommended)

Using `venv`:
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

Using `conda`:
```bash
conda create -n spam-classifier python=3.11
conda activate spam-classifier
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

The UCI SMS Spam Collection dataset is publicly available. Download it manually:

1. Visit: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Download the zip file
3. Extract `SMSSpamCollection` into the `data/` folder

Or via terminal:
```bash
cd data/
curl -L -o sms-spam-collection-dataset.zip https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset
unzip sms-spam-collection-dataset.zip
mv SMSSpamCollection spam.csv
```

**What the dataset looks like:**
```
ham    Go until jurong point, crazy.. Available only in bugis...
ham    Ok lar... Joking wif u oni...
spam   Free entry in 2 a wkly comp to win FA Cup final tkts...
ham    U dun say so early hor... U c already then say...
spam   WINNER!! As a valued network customer you have been selected...
```

Each row has two fields: a label (`ham` or `spam`) and the message text.

### Step 5 — Train the Model

```bash
python src/train.py
```

This will:
1. Load and preprocess the dataset
2. Vectorise the text using TF-IDF
3. Train a Multinomial Naive Bayes classifier
4. Save the trained model to `model/spam_classifier.pkl`
5. Print training accuracy

### Step 6 — Evaluate the Model

```bash
python src/evaluate.py
```

Output will include accuracy, precision, recall, F1-score, and a confusion matrix.

### Step 7 — Classify New Messages

```bash
python src/predict.py "Congratulations! You have won a FREE iPhone. Click now."
```

Or interactively:
```bash
python src/predict.py
```

### Step 8 — Run the Jupyter Notebook (Optional)

For a fully interactive walkthrough:
```bash
jupyter lab
```
Then open `notebooks/spam_classifier_walkthrough.ipynb`.

---

## AI Red Teaming Analysis

> This section is the **security layer** that transforms this from a standard ML project into an AI security portfolio piece.

### Attack Surface 1 — Adversarial Evasion

**What it is:** Crafting spam messages that fool the classifier into labelling them as ham.

**How it works against TF-IDF + Naive Bayes:**

TF-IDF scores words based on frequency. If "FREE" and "WIN" are high-weight spam indicators, an attacker can:
- Replace letters with lookalikes: `Fr33`, `W1N`, `F.R.E.E`
- Insert whitespace: `F R E E`
- Use synonyms: "complimentary" instead of "free"
- Pad the message with legitimate-sounding ham words to dilute the spam score

**Red Team Takeaway:** This model is brittle against obfuscation. Any engagement testing AI spam/phishing filters should include character substitution and synonym attacks.

---

### Attack Surface 2 — Data Poisoning

**What it is:** An attacker with write access to the training pipeline injects maliciously labelled samples to degrade or manipulate the model.

**How it works:**
- Inject 50–100 spam messages labelled as `ham` into the training set
- The model learns that those spam patterns are "legitimate"
- Future spam using those patterns bypasses the filter

**Red Team Takeaway:** In any AI pipeline audit, the data ingestion path is as important as the model itself. Ask: *who can write to the training dataset? Is it version-controlled? Is label integrity verified?*

---

### Attack Surface 3 — Model Extraction / Inference

**What it is:** Repeatedly querying the model to reverse-engineer its decision boundary, understanding which words trigger spam classification.

**How to probe this model:**
```python
# Send single-word messages and observe confidence scores
probe_words = ["free", "win", "urgent", "congratulations", "click", "claim"]
for word in probe_words:
    result = classifier.predict_proba([vectoriser.transform([word])])
    print(f"{word}: {result}")
```

By mapping high-confidence spam triggers, an attacker builds a "word blacklist" to avoid in future messages.

**Red Team Takeaway:** Classification models with exposed APIs should rate-limit queries and monitor for systematic probing patterns.

---

### Attack Surface 4 — Concept Drift Exploitation

**What it is:** Over time, spam tactics evolve. A model trained on 2011 data (this dataset) will not recognise 2025 spam patterns — crypto scams, AI-generated phishing, QR code lures.

**Red Team Takeaway:** Assess when the model was last retrained. Stale models are trivially bypassed with modern attack patterns.

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~98.2% |
| Precision | ~97.8% |
| Recall    | ~94.6% |
| F1-Score  | ~96.2% |

*Results may vary slightly depending on train/test split.*

---

## 🎓 Key Concepts Explained

### What is TF-IDF?

**TF-IDF** stands for Term Frequency-Inverse Document Frequency.

Think of it like this: if the word "FREE" appears 5 times in a single spam message but almost never in normal messages — it gets a HIGH score. If the word "the" appears everywhere — it gets a LOW score. TF-IDF rewards words that are specific and distinctive, not common filler words.

### What is Naive Bayes?

Naive Bayes calculates the **probability** that a message is spam, given the words it contains. It's "naive" because it assumes each word is independent — it doesn't understand that "you've won" together is more spammy than each word alone. Despite this simplification, it performs remarkably well on text classification tasks.

---

## 🔗 References

- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- [HTB Academy — Applications of AI in InfoSec](https://academy.hackthebox.com/module/292)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Adversarial ML Threat Matrix — MITRE ATLAS](https://atlas.mitre.org/)

---

## 📜 Disclaimer

This project is for **educational purposes only**, as part of an AI security learning path. All dataset usage complies with UCI's open-access terms.
