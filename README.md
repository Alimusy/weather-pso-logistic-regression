# PSO Feature Selection with Logistic Regression for Rainfall Prediction

Next-day rainfall prediction on the Rain in Australia (weatherAUS) dataset,
comparing Logistic Regression on all features against Logistic Regression on a
PSO-selected subset.

## Open it

PSO search and evaluation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alimusy/weather-pso-logistic-regression/blob/main/notebooks/pso_lr_corrected.ipynb)

Deploy the app yourself, one click, free:

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Alimusy/weather-pso-logistic-regression&branch=main&mainModule=app/app.py)

## Method

1. 22 features after preprocessing; PSO searches subsets using classifier
   performance as the fitness function and settles on 17.
2. SMOTE applied strictly inside cross-validation training folds.
3. Both configurations evaluated on accuracy, precision, recall, F1 and AUC-ROC.

## Results

| Model | Features | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| Baseline LR | 22 | 78.82% | 51.87% | 76.43% | 61.80% | 86.39% |
| PSO + LR | 17 | 78.83% | 51.89% | 76.17% | 61.73% | 86.32% |

Every gap is smaller than the run-to-run variation, so PSO buys a 23% smaller
feature set at no measurable cost in predictive quality.

## Repo layout

```
notebooks/   PSO search, training, evaluation
scripts/     standalone python versions of the pipeline
app/         Streamlit app
artifacts/   fitted pipelines, scaler, selected features, PSO cost history
figures/     chapter four figures (class balance, confusion matrices, convergence)
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Dataset

Rain in Australia (weatherAUS.csv) from Kaggle. Not committed.
