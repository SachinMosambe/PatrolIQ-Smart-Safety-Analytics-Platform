# 🚔 PatrolIQ - Smart Safety Analytics Platform

**PatrolIQ** is an end-to-end machine learning and data visualization platform for analyzing crime patterns in Chicago. It uses unsupervised learning techniques (K-Means, DBSCAN, Hierarchical Clustering) combined with dimensionality reduction (PCA, t-SNE, UMAP) and deployed with Streamlit for interactive exploration.

## 📋 Project Overview

### Goals
- **Geographic Hotspot Detection**: Identify crime clusters across Chicago using multiple clustering algorithms
- **Temporal Pattern Analysis**: Discover crime trends by hour, day, and month
- **Dimensionality Reduction**: Compress high-dimensional features using PCA and t-SNE for visualization
- **Experiment Tracking**: Log all models and metrics with MLflow
- **Interactive Dashboards**: Deploy a Streamlit web app for real-time exploration

### Key Features
✅ Up to 500K+ crime records from Chicago Data Portal  
✅ K-Means, DBSCAN, and Hierarchical clustering  
✅ PCA & t-SNE visualizations  
✅ MLflow experiment tracking  
✅ Streamlit interactive dashboard  
✅ Geographic heatmaps & temporal patterns  
✅ Arrest rate analysis by crime type  

## 📊 Dataset

**Source**: [Chicago Data Portal - Crime Dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

**Available Features**:
- Crime type and primary description
- Date & time (hour, day, month)
- Location (latitude, longitude, district, block)
- Arrest status
- Domestic violence flag
- FBI crime code

**Data Range**: 2001-Present (500K+ records)

## 🏗️ Project Structure

```
PatrolIQ Smart Safety Analytics Platform/
│
├── 📄 app.py                          # Streamlit dashboard (6 pages)
├── 📄 app_new.py                      # Alternative Streamlit version
├── 📄 optimize_data.py                # Data preprocessing script
├── 📄 requirements.txt                # Python dependencies
│
├── 📁 Notebooks/
│   ├── PatrolIQ_Full_Analysis.ipynb  # Main analysis notebook
│   ├── EDA.ipynb                      # Exploratory Data Analysis
│   ├── feature_engineering.ipynb      # Feature creation & ML
│   ├── preprocessing.ipynb            # Data cleaning
│   └── Notebook.ipynb                 # Additional analysis
│
├── 📁 Data/
│   ├── app_crime_data.csv             # Streamlit dataset (10K sample)
│   ├── processed_crime_data.csv       # Processed full dataset
│   ├── clean_crime_data.csv           # Cleaned dataset
│   └── Crimes_-_2001_to_Present_*.csv # Raw Chicago data export
│
├── 📁 models/
│   └── tsne_embeddings.npy            # Pre-computed t-SNE projections
│
├── 📁 mlruns/                         # MLflow experiment tracking
│
└── 📄 README.md                       # This file
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd "PatrolIQ Smart Safety Analytics Platform"
python -m venv venv
venv\scripts\activate
pip install -r requirements.txt
```

### 2. Run Analysis Notebook
```bash
jupyter notebook Notebooks/PatrolIQ_Full_Analysis.ipynb
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📈 Streamlit Dashboard Pages

### 📊 **Overview**
- KPI metrics (total crimes, arrest rate, crime types)
- Top 10 crime types distribution
- Arrest rates by crime type
- Top districts by crime count
- Hourly crime trends

### 🗺️ **Geographic Hotspots**
- Interactive mapbox visualization
- K-Means cluster overlay
- Cluster statistics (crime count, arrest rate)
- Top crime types per cluster

### ⏰ **Temporal Patterns**
- Hourly distribution chart
- Day-of-week comparison
- Weekend vs. weekday analysis
- Crime heatmap (hour × day)
- Monthly trend line

### 🔬 **Dimensionality Reduction**
- **PCA Analysis**: 2D projection with variance explained
- **t-SNE Visualization**: 
  - Color by crime type
  - Color by geographic cluster
  - Color by time of day
- Feature importance scores

### 📈 **Model Performance**
- Model status (loaded/failed)
- Algorithm parameters
- Silhouette & Davies-Bouldin scores
- MLflow dashboard link

### 🔍 **Crime Analysis**
- Arrest rates by crime type
- Top crime locations (pie chart)
- Detailed crime records table
- Filters for crime types & districts

## 🤖 Machine Learning Algorithms

### Clustering Methods

| Algorithm | Use Case | Strengths |
|-----------|----------|-----------|
| **K-Means** | Geographic hotspots | Fast, scalable, clear boundaries |
| **DBSCAN** | Density-based zones | Finds arbitrary shapes, identifies noise |
| **Hierarchical** | Dendrogram analysis | Flexible number of clusters |

### Dimensionality Reduction

| Technique | Purpose | Output |
|-----------|---------|--------|
| **PCA** | Linear compression | 2D projection, variance explained |
| **t-SNE** | Nonlinear visualization | 2D embedding, preserves clusters |
| **UMAP** | Fast alternative to t-SNE | 2D/3D embedding (optional) |

### Evaluation Metrics

- **Silhouette Score**: Measures cluster cohesion (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Cluster separation metric (lower is better)
- **Elbow Method**: Optimal K selection for K-Means

## 📊 Key Findings (Example)

From a sample analysis of 50K records:

- **Top Crime Types**: THEFT, BATTERY, CRIMINAL DAMAGE, BURGLARY
- **Peak Hours**: 12-14 (noon-2 PM) and 18-22 (evening)
- **Geographic Clusters**: 5-10 major hotspots identified
- **Arrest Rates**: 25-35% overall, varies by crime type
- **Weekend Effect**: 5-10% higher crime on weekends

## 🔧 Configuration & Customization

### MLflow Tracking
Edit `app.py` to set your MLflow tracking URI:
```python
DAGSHUB_REPO = "YourUsername/PatrolIQ-..."
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO}.mlflow")
```

### Data Filtering
Modify `app.py` to change default filters:
```python
crime_filter = st.sidebar.multiselect(
    "Crime Types",
    crime_types,
    default=crime_types  # Show all by default
)
```

### Clustering Parameters
Adjust in `PatrolIQ_Full_Analysis.ipynb`:
```python
best_k_range = range(5, 11)  # Change K range
km = KMeans(n_clusters=k, random_state=42, n_init=10)
```

## 📦 Dependencies

**Core Libraries**:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML algorithms
- `matplotlib`, `seaborn` - Static visualizations
- `plotly` - Interactive charts

**Deployment & Tracking**:
- `streamlit` - Web app framework
- `mlflow` - Experiment tracking
- `joblib` - Model serialization

**Optional**:
- `umap-learn` - Advanced dimensionality reduction
- `jupyter` - Notebook environment
- `requests` - API calls

See `requirements.txt` for exact versions.

## 🔄 Data Pipeline

```
Raw Data (Chicago API)
    ↓
Data Cleaning & Validation
    ↓
Feature Engineering (hour, day, month, etc.)
    ↓
Standardization & Scaling
    ↓
Clustering (K-Means, DBSCAN, Hierarchical)
    ↓
Dimensionality Reduction (PCA, t-SNE)
    ↓
MLflow Logging
    ↓
Streamlit Dashboard
```

## 🎯 Use Cases

1. **Police Resource Allocation**: Identify high-crime areas for increased patrols
2. **Public Safety Planning**: Discover temporal patterns (late night, weekends)
3. **Risk Assessment**: Cluster neighborhoods by crime severity
4. **Community Awareness**: Visualize safety patterns for residents
5. **Academic Research**: Unsupervised learning case study

## 📚 References & Resources

- [Chicago Data Portal API](https://data.cityofchicago.org/developers)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [MLflow Documentation](https://mlflow.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [t-SNE Paper](https://jmlr.org/papers/v9/vandermaaten08a.html)

## 👤 Author & License

**Creator**: Sachin Mosambe  
**Repository**: [GitHub PatrolIQ](https://github.com/SachinMosambe/PatrolIQ-Smart-Safety-Analytics-Platform)  
**License**: MIT

## 🐛 Troubleshooting

### Issue: Models not loading
**Solution**: Check MLflow run IDs in `app.py` match your actual runs:
```bash
mlflow ui --backend-store-uri mlruns
```

### Issue: t-SNE not available
**Solution**: Run `PatrolIQ_Full_Analysis.ipynb` first to generate embeddings:
```bash
jupyter notebook Notebooks/PatrolIQ_Full_Analysis.ipynb
```

### Issue: Out of memory on large datasets
**Solution**: Reduce sample size in notebooks or use data streaming

### Issue: Streamlit performance slow
**Solution**: Clear cache with `streamlit cache clear` or use filtered datasets

## 📝 Citation

If you use PatrolIQ in your research:

```bibtex
@project{PatrolIQ2024,
  title={PatrolIQ: Smart Safety Analytics Platform},
  author={Mosambe, Sachin},
  year={2024},
  url={https://github.com/SachinMosambe/PatrolIQ-Smart-Safety-Analytics-Platform}
}
```

## 🙌 Contributing

Contributions welcome! Areas for improvement:
- Real-time data ingestion
- Predictive models (LSTM, Prophet)
- Additional clustering metrics
- Mobile app version
- API endpoint creation

---

**Built with ❤️ for smart city analytics**  
Last updated: November 2024
