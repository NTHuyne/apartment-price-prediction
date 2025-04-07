# apartment-price-prediction
## About The Project

A data pipeline to collect data from various sources, process the data, and predict apartment prices in Hanoi using machine learning and deep learning models.

**Data Sources**: The project collects data from alonhadat, bds68, and homedy.

**Models Used**: The project uses Linear Regression, SVR, and XGBoost models.

## Getting started

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/NTHuyne/apartment-price-prediction.git
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
## Running the Code

1.  Clone the repository.
2.  Install the dependencies using `pip install -r requirements.txt`.
3.  Run the `src/modeling/model_training.py` file to train and save the models.
4.  To run the demo app, install Streamlit using `pip install streamlit` and then run the `app/interface.py` file using `streamlit run app/interface.py`.

## File Structure

-   `data/`: Contains the raw, normalized, and finalized datasets.
-   `model/`: Contains the feature engineering preprocessor and the trained models.
-   `src/`: Contains the source code for data collection, data processing, feature engineering, modeling, and evaluation.
-   `app/`: Demo interface with Streamlit

## Contributing
| Name                   | Student ID | 
|------------------------|------------|
| Trịnh Giang Nam        | 20215229   | 
| Nguyễn Trọng Huy       | 20210451   | 
| Nguyễn Chính Minh      | 20215224   |
