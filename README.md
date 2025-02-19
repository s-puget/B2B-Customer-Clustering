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

## Clustering Notebook  

The main notebook of the project is **clustering.ipynb**. This notebook serves as the core of the analysis, bringing together processed data from previous steps. It merges `client_summary_segmentation.csv` and `client_transaction_product_file.csv` on `client_id` to create a unified dataset.  

After conducting exploratory data analysis (EDA) and additional feature engineering, the focus shifts to developing and refining clustering models. The notebook evaluates different clustering techniques, interprets the results, and provides insights into client segmentation.  

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
│── 📁 src  
│   ├── `data_loader.py` - Loads and merges datasets.  
│   ├── `preprocessing.py` - Cleans and encodes data.  
│   ├── `feature_engineering.py` - Implements parameter weighting.  
│   ├── `clustering.py` - Applies DBSCAN for clustering.  
│   ├── `evaluation.py` - Evaluates clustering results.
├── `clustering.ipynb`  
│── `presentation.pdf`  
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


