# WLDA - Weighted missing Linear Discriminant Analysis
The code repository associated with the paper: **"Interpretable Linear Discriminant Analysis in the presence of missing data"**
This paper is under evaluation for the journal you can find it ....

## 1. Introduction
Weighted missing Linear Discriminant Analysis (WLDA): an algorithm can effectively handle missing data without the need for imputation methods by employing a weighted missing matrix to assess the contribution of each feature with missing values, ensuring fair treatment across features. The WLDA algorithm not only enhances the robustness and accuracy of models in the presence of missing data but also ensures that the decision-making process is transparent and comprehensible. This dual focus on data integrity and explainability is crucial for fostering trust, reliability, and broader acceptance of AI technologies in critical applications.

## 2. Repository Contents
The codes are structured as follows:
```
.
├── src                    
│   ├── results               # Directory for storing experimental results
│   ├── __init__.py           
│   ├── dper.py           
│   ├── experiment.py
│   ├── funcs.py
│   ├── loaddata.py
│   ├── main.py
│   ├── mylda.py
│   ├── plot.py
│   ├── shapvalues.py
│   ├── showresults.py           
│   └── wlda.py             
└── README.md
```
In `/src` folders: 
- `mylda.py` contains the implementation of the LDA (Linear Discriminant Analysis) algorithm.
- `wlda.py` contains the implementation of the WLDA (Weighted Linear Discriminant Analysis) algorithm.
- `dpers.py` implements the DPER algorithm for computing the covariance matrix used in the WLDA.
- `shapvalues.py` calculates Shapley values. Shapley values are a method from cooperative game theory applied in machine learning to explain the prediction of a model by attributing the outcome to different features.
