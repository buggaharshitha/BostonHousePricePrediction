# Boston Housing Price Prediction

A machine learning project that predicts Boston housing prices using the classic Boston Housing dataset.  
The project includes data analysis, model training, a REST API backend (Flask), and an interactive web UI (Streamlit).

---

## Project Structure

```
project/
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
│── train_model.py            # ML model training script
│── app.py                    # Backend REST API using Flask
│── ui.py                     # Frontend UI using Streamlit
│── Harshitha_BostonHousing.ipynb  # Notebook with EDA & analysis
│── report.docx               # Project report
│── data/
│   └── boston_housing.csv    # Dataset
│── outputs/                  # Charts, model file, risk tables
```

---

## Dataset Features

| Feature   | Description                                      |
|-----------|--------------------------------------------------|
| crim      | Per capita crime rate by town                    |
| zn        | Proportion of residential land zoned > 25k sqft  |
| indus     | Proportion of non-retail business acres          |
| chas      | Charles River dummy variable (1 if bounds river) |
| nox       | Nitric oxide concentration (parts per 10M)       |
| rm        | Average number of rooms per dwelling             |
| age       | Proportion of owner-occupied units built pre-1940|
| dis       | Weighted distances to five Boston employment centres |
| rad       | Index of accessibility to radial highways        |
| tax       | Full-value property-tax rate per $10,000         |
| ptratio   | Pupil-teacher ratio by town                      |
| b         | 1000(Bk - 0.63)^2 where Bk is Black proportion  |
| lstat     | % lower status of the population                 |
| **medv**  | **Median home value in $1000s (target)**         |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (saves model to outputs/)
python train_model.py

# 3. Start the Flask API
python app.py

# 4. Launch the Streamlit UI (in a separate terminal)
streamlit run ui.py
```

---

## API Usage

**Endpoint:** `POST /predict`  
**Content-Type:** `application/json`

```json
{
  "crim": 0.00632, "zn": 18.0, "indus": 2.31, "chas": 0,
  "nox": 0.538, "rm": 6.575, "age": 65.2, "dis": 4.09,
  "rad": 1, "tax": 296.0, "ptratio": 15.3, "b": 396.9, "lstat": 4.98
}
```

**Response:**
```json
{ "predicted_price": 24.35 }
```

---

## Model

- **Algorithm:** Random Forest Regressor  
- **Evaluation metrics:** RMSE, MAE, R²  
- **Model artifact:** `outputs/model.pkl`

---

## Results

Charts and outputs are saved in the `outputs/` directory after running `train_model.py`.

---

## Author

Harshitha — Boston Housing Price Prediction Project
