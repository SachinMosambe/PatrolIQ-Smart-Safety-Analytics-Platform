Below is a **clean, professional, perfectly structured README.md** — rewritten from your version, simplified, corrected, and formatted for GitHub.
Your AWS/MLflow and project documentation is now **clear, concise, and production-ready**.

---

# ✅ **Clean & Professional README.md for PatrolIQ**

````markdown
# 🚔 PatrolIQ – Smart Safety Analytics Platform

PatrolIQ is an end-to-end machine learning and data visualization platform for analyzing crime patterns in Chicago. The project uses clustering, dimensionality reduction, MLflow for experiment tracking, and Streamlit for interactive dashboards.

---

# 🚀 MLflow Deployment on AWS (Production Setup)

Follow these steps to deploy MLflow on AWS so your models load correctly from S3 & Streamlit Cloud.

---

## ✅ 1. Create AWS Resources

### **1️⃣ IAM User**
- Create an IAM user with **Programmatic access**
- Attach policy:  
  ✔ `AmazonS3FullAccess`  
  ✔ `AmazonRDSFullAccess`  
  ✔ `AmazonEC2FullAccess` *(optional)*  

Save:
- Access Key ID  
- Secret Access Key  

### **2️⃣ Configure AWS CLI on EC2**
```bash
aws configure
````

Enter:

* AWS Access Key
* AWS Secret Key
* Region: `ap-south-1`

### **3️⃣ Create S3 Bucket**

Example:

```
mlflow-tracking-bucket46
```

### **4️⃣ Create EC2 Instance**

* Ubuntu (t2.large recommended)
* Open security group port:

| Port | Purpose   |
| ---- | --------- |
| 5000 | MLflow UI |
| 22   | SSH       |
| 80   | Optional  |

---

## ✅ 2. Install MLflow on EC2

```bash
sudo apt update
sudo apt install python3-pip -y
sudo apt install python3.12-venv -y

mkdir mlflow && cd mlflow
python3 -m venv venv
source venv/bin/activate
pip install mlflow boto3 awscli psycopg2-binary
```

---

## ✅ 3. Set MLflow to use S3

Start server:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://postgres:<PASSWORD>@<RDS-ENDPOINT>:5432/mlflow \
  --default-artifact-root s3://mlflow-tracking-bucket46 \
  --allowed-hosts="*"
```

Open MLflow in browser:

```
http://<EC2-PUBLIC-IP>:5000
```

---

## ✅ 4. Set Tracking URI (Local or Streamlit Cloud)

### Local (Mac/Windows/EC2)

```bash
export MLFLOW_TRACKING_URI=http://<EC2-PUBLIC-IP>:5000
```

### Streamlit Cloud

Add in **Settings → Secrets**:

```toml
MLFLOW_TRACKING_URI = "http://<EC2-PUBLIC-IP>:5000"

AWS_ACCESS_KEY_ID = "YOUR_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET"
AWS_DEFAULT_REGION = "ap-south-1"

S3_BUCKET = "mlflow-tracking-bucket46"
MODEL_KEY = "YOUR/MODEL/PATH/model.pkl"
```

❗Streamlit Cloud CANNOT read local exports — **must use secrets**.

---

# 📊 PatrolIQ Features

### ✔ Geographic Crime Hotspots

* K-Means
* DBSCAN
* Hierarchical clustering
* Interactive PyDeck & Plotly maps

### ✔ Temporal Crime Patterns

* Hourly analysis
* Day-of-week patterns
* Weekend vs weekday
* Heatmaps

### ✔ Dimensionality Reduction

* PCA (variance explained)
* t-SNE & UMAP visualizations

### ✔ MLflow Tracking

* Compare clustering algorithms
* Silhouette & Davies-Bouldin scores
* Registered models & runs

### ✔ Streamlit Dashboard

* 6-page interactive UI
* Filters (year, crime type, district)
* Summary metrics & visual analytics

---

# 📂 Project Structure

```
PatrolIQ/
│
├── app.py                     # Main Streamlit app
├── app_new.py                 # Alternate Streamlit version
├── optimize_data.py           # Data preprocessing
├── requirements.txt           # Dependencies
│
├── Notebooks/
│   ├── PatrolIQ_Full_Analysis.ipynb
│   ├── feature_engineering.ipynb
│   ├── preprocessing.ipynb
│   └── EDA.ipynb
│
├── Data/
│   ├── app_crime_data.csv
│   ├── clean_crime_data.csv
│   └── processed_crime_data.csv
│
├── models/
│   └── tsne_embeddings.npy
│
└── README.md
```

---

# 🔧 Quick Start (Local)

```bash
git clone https://github.com/SachinMosambe/PatrolIQ-Smart-Safety-Analytics-Platform.git
cd PatrolIQ-Smart-Safety-Analytics-Platform

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

---

# 📈 Visualizations in Dashboard

### 📊 Overview

* Crime KPIs
* Monthly trends
* Geographic heatmaps
* Top crime types

### 🗺 Clustering

* Cluster maps
* Cluster statistics
* Hotspot zones

### ⏰ Temporal

* Hourly/weekly trends
* Heatmaps
* Crime pattern clusters

### 🔬 Dimensionality Reduction

* PCA
* t-SNE
* Feature importance

### 🎯 Model Performance

* Metrics tables
* MLflow links
* Comparison bar charts

---

# 🤖 Machine Learning Used

| Category                 | Algorithms                    |
| ------------------------ | ----------------------------- |
| Clustering               | K-Means, DBSCAN, Hierarchical |
| Dimensionality Reduction | PCA, t-SNE, UMAP              |
| Metrics                  | Silhouette Score, DB Index    |

---

# 🐛 Troubleshooting

### ❌ Streamlit Cloud Error: NoCredentialsError

Fix → add AWS keys in **Streamlit Secrets**.

### ❌ MLflow Not Loading

Fix → ensure tracking URI points to EC2:

```bash
export MLFLOW_TRACKING_URI=http://<EC2-PUBLIC-IP>:5000
```

### ❌ Dataset too large

Fix → use `clean_crime_data.csv` (smaller processed file).

---

# 👤 Author

**Sachin Mosambe**
GitHub: [https://github.com/SachinMosambe](https://github.com/SachinMosambe)

---

# 🌍 Live Demo

👉 **Streamlit Cloud App:**
[https://patroliq-smart-safety-analytics-platform-yrsksqspjudecgyidjc3d.streamlit.app/](https://patroliq-smart-safety-analytics-platform-yrsksqspjudecgyidjc3d.streamlit.app/)

---




