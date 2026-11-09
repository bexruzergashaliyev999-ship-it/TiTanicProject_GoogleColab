# Titanic Survival Prediction

A beginner-friendly machine learning project that predicts whether a passenger survived the Titanic disaster. It walks through a complete workflow in a single Jupyter notebook: loading data, exploring it, cleaning it, training a model and evaluating the result.

The notebook is `TITANIC.ipynb`.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bexruzergashaliyev999-ship-it/TiTanicProject_GoogleColab/blob/main/TITANIC.ipynb)
---

## Highlights

- Explores the classic Titanic dataset (**891 passengers, 12 columns**)
- Visualizes survival by sex, ticket class and age
- Cleans missing values and converts text columns to numbers
- Trains a **Logistic Regression** classifier
- **81% accuracy** on the test set (145 of 179 passengers predicted correctly)
- Predicts the survival probability of a custom passenger

---

## Dataset

- **Source**: [Data Science Dojo Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Size**: 891 rows x 12 columns
- **Target**: `Survived` (0 = did not survive, 1 = survived)
- The notebook downloads the data automatically, so an internet connection is required.

| Column | Description |
|--------|-------------|
| `Survived` | Survival (0 / 1), the target |
| `Pclass` | Ticket class (1, 2 or 3) |
| `Sex` | Passenger sex |
| `Age` | Age in years |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Fare` | Ticket price |
| `Embarked` | Port of embarkation (C, Q or S) |
| `PassengerId`, `Name`, `Ticket`, `Cabin` | Dropped during cleaning |

---

## Workflow

### Step 1: Load the data
Import the libraries and load the CSV with pandas.

### Step 2: Inspect the data
Use `info()` and `isnull().sum()` to check column types and missing values:

| Column | Missing values |
|--------|---------------|
| `Age` | 177 |
| `Cabin` | 687 |
| `Embarked` | 2 |

### Step 3: Explore with charts
- Survival rate by sex
- Survival rate by ticket class
- Age distribution of survivors vs. non-survivors

### Step 4: Drop unneeded columns
`PassengerId`, `Name`, `Ticket` and `Cabin` are removed.

### Step 5: Clean the data
- Fill missing `Age` values with the median age
- Fill the 2 missing `Embarked` values with the most common port
- Convert `Sex` to numbers (male = 0, female = 1)
- One-hot encode `Embarked` with `pd.get_dummies` (`drop_first=True`)

### Step 6: Train and evaluate the model
- 80% train / 20% test split (`random_state=42`)
- `LogisticRegression` from scikit-learn
- Results:

| Metric | Value |
|--------|-------|
| Training score | 0.801 |
| Test score (accuracy) | 0.810 |
| Correct predictions | 145 |
| Wrong predictions | 34 |

### Step 7: Predict for a custom passenger
The final cell builds a hypothetical passenger and prints the model's survival probability with `predict_proba`.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data loading and processing |
| matplotlib | Charts |
| scikit-learn | Train/test split and Logistic Regression |
| Jupyter | Interactive notebook |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook TITANIC.ipynb
```

Run the cells from top to bottom. The notebook also works in Google Colab (upload it and use **Runtime → Run all**).

---

## Project Structure

```
.
├── TITANIC.ipynb        # Full analysis
├── requirements.txt
└── README.md
```

---

## Possible Improvements

- Increase `max_iter` or scale the features to remove the Logistic Regression convergence warning
- Apply exactly the same preprocessing to the training data and to custom passengers (for example, the `Fare` transformation used in the prediction cell is not applied during training)
- Add engineered features such as family size, "is alone" and title extracted from the name
- Compare other models (Random Forest, Gradient Boosting, SVM) and use cross-validation
- Evaluate with precision, recall, F1 score and a confusion matrix, not only accuracy

---

## Acknowledgements

- Dataset by [Data Science Dojo](https://github.com/datasciencedojo/datasets)
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/) and [matplotlib](https://matplotlib.org/)

---

## License

This project is licensed under the MIT License. The dataset belongs to its original owners and is subject to their terms.
