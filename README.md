# Client Clustering Project

This project was conducted as part of a school collaboration with a company specializing in electrical supplies distribution. The goal was to **perform client clustering** by **analyzing data from multiple sources**, including client information, website interactions, product details, and transaction records.

## Project Overview

We worked with four main datasets:
- **Client Information**: Contains general details about clients.
- **Website Interaction Data**: Tracks client behavior on the company’s website.
- **Product Information**: Details about the products available.
- **Transaction Data**: Records of client purchases.

The workflow involved:
1. **Exploratory Data Analysis (EDA)**: Understanding each dataset individually.
2. **Feature Engineering**: Creating meaningful features to improve clustering.
3. **Merging Data**: Integrating multiple tables to form a comprehensive client dataset.
4. **Clustering Analysis**: Applying machine learning techniques to segment clients.

## Notebooks

The project consists of four main Jupyter notebooks:

1. **transactions_exploration.ipynb**  
   - EDA on transaction data.  

2. **google_analytics_exploration.ipynb**  
   - EDA and feature engineering on website interaction data.  
   - Outputs a processed dataset: `client_summary_segmentation.csv`.

3. **client_product_transaction_exploration.ipynb**  
   - EDA and feature engineering on client, product, and transaction data.  
   - Outputs a processed dataset: `client_transaction_product_file.csv`.

4. **clustering.ipynb** (Main focus of my contribution)  
   - Merges `client_summary_segmentation.csv` and `client_transaction_product_file.csv` on `client_id` to create a unified dataset.  
   - Performs EDA and additional feature engineering.  
   - Develops and refines clustering models.  
   - Evaluates and interprets clustering results.

## Clustering Methodology

The clustering process included:
- **Preprocessing**: Handling missing values, scaling, and encoding categorical features.
- **Feature Engineering**: Creating new variables to enhance segmentation.
- **Parameter Weighting**: Adjusting feature importance to refine clustering results.
- **Modeling**: Applying algorithms such as K-Means, DBSCAN, or hierarchical clustering.
- **Evaluation**: Assessing clusters using metrics like silhouette score, Davies-Bouldin Index, and Calinski-Harabasz Index.
- **Refinement**: Improving clustering performance by optimizing parameters.
- **Visualization**: Using **PCA** to visualize clusters.

## Repository Structure

📂 B2B-Customer-Clustering  
│── 📁 data  
│   ├── `client_summary_segmentation.csv`  
│   ├── `client_transaction_product_file.csv`  
│── 📁 notebooks  
│   ├── `transactions_exploration.ipynb`  
│   ├── `google_analytics_exploration.ipynb`  
│   ├── `client_product_transaction_exploration.ipynb`  
│   ├── `clustering.ipynb`  
│── 📁 src  
│   ├── `data_loader.py` - Loads and merges datasets.  
│   ├── `preprocessing.py` - Cleans and encodes data.  
│   ├── `feature_engineering.py` - Implements parameter weighting.  
│   ├── `clustering.py` - Applies DBSCAN for clustering.  
│   ├── `evaluation.py` - Evaluates clustering results.  
│── `requirements.txt`  
│── `README.md`

## How to Run the Code

1. **Set up the environment**
Ensure you have Python and required libraries installed. You can install dependencies using:
  ```bash
  pip install -r requirements.txt
```
2. **Run the pipeline**
  Execute:
  ```bash
  python src/main.py
```
This will generate a clustered dataset and output evaluation metrics and PCA vizualisation graph.


