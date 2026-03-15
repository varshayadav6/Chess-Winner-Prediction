# ♟️ Chess Winner Prediction

A Machine Learning project that predicts the probability of winning a chess game based on player ratings, opening moves, and game statistics.

---

## 📌 Project Overview

Chess is a game of intelligence, strategy, and complexity. This project analyzes over **20,000 chess games** to build a predictive model that determines the probability of the **white player winning** based on various game features like first moves, player ratings, and opening strategies.

---

## 📂 Dataset

- **Source:** Lichess online chess games dataset
- **Size:** 20,058 records × 16 columns
- **Target Variable:** `winner` (white / black / draw)

### Key Features Used:
| Feature | Description |
|---|---|
| `rated` | Whether the game is rated or casual |
| `turns` | Total number of turns in the game |
| `white_rating` | ELO rating of the white player |
| `black_rating` | ELO rating of the black player |
| `opening_eco` | Encyclopedia of Chess Openings code |
| `opening_ply` | Number of moves in the opening phase |
| `white_1move` | First move of the white player |
| `black_1move` | First move of the black player |

---

## 🔧 Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Pickle

---

## 🚀 Project Workflow

### 1. Data Understanding
- Explored dataset shape, columns, data types
- Visualized winner distribution, victory status, ratings

### 2. Data Preprocessing
- Dropped unnecessary columns: `id`, `created_at`, `last_move_at`, `increment_code`, `white_id`, `black_id`, `opening_name`
- Extracted **first moves** of both white and black players from the `moves` column
- Mapped first moves to their **winning probability**
- Encoded categorical columns (`rated`, `winner`) to numeric values
- Handled 18 missing values in `black_1move` by dropping those rows

### 3. Model Building
Three classification models were trained and evaluated:

| Model | Accuracy | ROC-AUC Score |
|---|---|---|
| Logistic Regression | 66% | - |
| Support Vector Machine (SVC) | 65% | - |
| **Random Forest** ✅ | **69%** | **0.758** |

### 4. Hyperparameter Tuning
Used **GridSearchCV** to tune the Random Forest model.

Best Parameters:
```
criterion='entropy', max_features=4, min_samples_split=4
```

### 5. Final Model Performance
- **Accuracy:** 69%
- **Cross Validation Score:** 69%
- **ROC-AUC Score:** 75.7%

### 6. Model Saving
The final model is saved using **Pickle** for future predictions.

---

## 📁 Project Structure

```
Chess Winner Prediction Project/
│
├── games.csv                          # Dataset
├── chess_winner_prediction.ipynb      # Main Jupyter Notebook
└── README.md                          # Project Documentation
```

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/varshayadav6/chess-winner-prediction.git
```

2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the Jupyter Notebook:
```bash
jupyter notebook chess_winner_prediction.ipynb
```

4. Run all cells from top to bottom.

---

## 📊 Key Insights

- **White player wins more often** (~50%) compared to black (~45%) and draw (~5%)
- **Resign** is the most common victory status (~55%)
- Player **ratings** are strong predictors of the game outcome
- The **Random Forest** model outperformed Logistic Regression and SVM

---

## 💾 Load & Use Saved Model

```python
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open("chess_winner_prediction.pickle", "rb"))

# Predict for new data (8 features)
data = np.array([[1, 45, 1500, 1450, 0.51, 3, 0.50, 0.49]])
prediction = model.predict(data)

print("White Wins!" if prediction[0] == 1 else "Black Wins / Draw")
```

---

## 👤 Author

**Kondi Varsha**  
[GitHub](https://github.com/Kondi-Varsha) | [LinkedIn](https://www.linkedin.com/in/kondi-varsha-22-08-2003-yadav/)
