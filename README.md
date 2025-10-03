# ⚡ Client Clustering Project

This project was conducted as part of a school collaboration with a company specializing in electrical supplies distribution. The goal was to **perform client clustering** by analyzing data from multiple sources, including client information, website interactions, product details, and transaction records.

## 🗂 Project Overview

We worked with four main datasets:  
- **Client Information**: General details about clients.  
- **Website Interaction Data**: Tracks client behavior on the company’s website.  
- **Product Information**: Details about available products.  
- **Transaction Data**: Records of client purchases.  

The workflow involved:  
1. 🔍 **Exploratory Data Analysis (EDA)**: Understanding each dataset individually.  
2. 🛠 **Feature Engineering**: Creating meaningful features to improve clustering.  
3. 🔗 **Data Integration**: Merging datasets to form a comprehensive client dataset.  
4. 📊 **Clustering Analysis**: Applying machine learning techniques to segment clients.

## 👤 My Role

As part of a group project, tasks were divided among team members. My responsibilities included:  
- Merging preprocessed data from classmates into a unified dataset.  
- Performing additional preprocessing and feature engineering.  
- Testing and refining different clustering approaches.  
- Evaluating models and optimizing parameters for better segmentation.  

All these steps are documented in the **clustering.ipynb** notebook. This notebook merges `client_summary_segmentation.csv` and `client_transaction_product_file.csv` on `client_id` to create a unified dataset, then proceeds with preprocessing, feature engineering, and clustering analysis.

## 🧩 Clustering Methodology

The clustering process included:  
- 🔧 **Preprocessing**: Handling missing values, scaling, and encoding categorical features.  
- 🛠 **Feature Engineering**: Creating new variables to enhance segmentation.  
- ⚖️ **Parameter Weighting**: Adjusting feature importance to refine results.  
- 🤖 **Modeling**: Applying algorithms such as K-Means, DBSCAN, or hierarchical clustering.  
- 📏 **Evaluation**: Assessing clusters using silhouette score, Davies-Bouldin Index, and Calinski-Harabasz Index.  
- 🔄 **Refinement**: Optimizing parameters to improve clustering performance.  
- 🎨 **Visualization**: Using PCA to visualize clusters and interpret patterns.
