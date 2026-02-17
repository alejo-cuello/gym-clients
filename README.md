# Gym Clients Analysis

## Project Overview
This project focuses on analyzing gym client data to show basic information about the clients behavior. By leveraging data analysis and machine learning techniques, we aim to segment clients based on their workout routines and engagement levels.

## Objectives
- **Data Cleaning & Preprocessing**: Transform raw data into a structured format suitable for analysis.
- **Exploratory Data Analysis (EDA)**: Gain insights into client demographics, routine frequency, and exercise preferences.
- **Client Clustering**: Segment clients into distinct groups using algorithms like K-Means, DBSCAN, and Hierarchical Clustering to identify behavioral patterns (e.g., consistent goers, casual visitors).
- **Show the data**: Show the gym owner the information gathered.

## **Personal Objectives**:
- **Apply more AI Developing Tools**: I used tools like Gemini CLI, Antigravity and Gemini Chat to accelerate the development process. This allowed me to focus more on the analysis.

## Project Structure
```
gym-clients/
├── data/                  # Data storage (on Github, only processed)
├── notebooks/             # Jupyter Notebooks for analysis
│   ├── 01_eda.ipynb       # Initial data cleaning and EDA
│   ├── 02_eda_gym_routines_by_month.ipynb # Time-series analysis of routines (implied)
│   └── 03_eda_clusters.ipynb # Client clustering analysis
├── scripts/               # Python modules for data processing and modeling
│   ├── data-preprocessing/
│   └── modeling/
├── .gitignore
├── LICENSE
└── README.md
```

## Key Analyses
1.  **Exploratory Data Analysis**:
    -   Data cleaning (normalizing text, handling missing values).
    -   Feature engineering (generating `client_id`, extracting temporal features).

2.  **Cluster Analysis**:
    -   Grouping clients based on `average_of_days_per_routine`,`routines_count`,`gender_encoded`,`tenure_months`,`recency_months`.
    -   Evaluation of clusters using K-Means, DBSCAN, and Hierarchical methods.


## License
View the [LICENSE](LICENSE) file for details.
