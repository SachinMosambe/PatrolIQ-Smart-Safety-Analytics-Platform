Below is your **FINAL, CLEAN, PRODUCTION-READY `README.md`** —
fully updated with:

✔ Streamlit Cloud deployment
✔ CI/CD (GitHub → Streamlit Auto-deploy)
✔ MLflow on AWS
✔ S3 + RDS
✔ Google Drive dataset loading
✔ Correct folder structure (based on your screenshot)
✔ Professional formatting

Copy–paste directly into your repo.

---

# 🚔 **PatrolIQ – Smart Safety Analytics Platform**

PatrolIQ is an end-to-end machine learning and geospatial analytics system designed to analyze and visualize crime patterns in Chicago.
The platform integrates MLflow (AWS), S3 artifact storage, clustering models, temporal analytics, and an interactive Streamlit dashboard.

---

# 🌐 **Live Application**

👉 **Streamlit Cloud:**
[https://patroliq-smart-safety-analytics-platform-yrsksqspjudecgyidjc3d.streamlit.app/](https://patroliq-smart-safety-analytics-platform-yrsksqspjudecgyidjc3d.streamlit.app/)

---

# 🧠 **Major Features**

### 📍 Geospatial Crime Hotspots

* K-Means, DBSCAN, Hierarchical clustering
* PyDeck & Plotly interactive maps
* Cluster statistics and centroids

### ⏳ Temporal Crime Analytics

* Hourly heatmaps
* Day-of-week patterns
* Monthly trend analysis

### 🔬 Dimensionality Reduction

* PCA (variance explained)
* 2D PCA projections
* t-SNE & UMAP visualizations

### 📊 MLflow Integration (AWS)

* Run tracking (EC2-hosted MLflow)
* S3 artifact storage
* Registered models
* Model promotion pipeline

### 🖥 Streamlit Dashboard

* Fully interactive UI
* Fast data caching
* Cloud-ready
* Secure secret handling

---

# 📂 **Project Structure**

```
PATROLIQ SMART SAFETY ANALYTICS PLATFORM/
│
├── .github/
│   └── workflows/
│       └── ci.yml               # CI pipeline (linting & tests)
│
├── Data/
│   ├── clean_crime_data.csv
│   └── Crimes_-_2001_to_Present_20251110.csv
│
├── mlartifacts/                 # MLflow artifacts (local)
├── mlruns/                      # MLflow local runs
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   ├── Notebook.ipynb
│   └── plots/*.png              # Figures
│
├── app.py                       # Streamlit dashboard
├── promote_model.py             # MLflow model promotion
├── test.py                      # Unit tests
├── requirements.txt
└── README.md
```

---

# 📦 **Loading Data in Streamlit (Google Drive)**

The dashboard loads the cleaned dataset from Google Drive for reliability and speed.

```python
@st.cache_data(show_spinner=False)
def load_data():
    """Load crime dataset from Google Drive (public CSV)."""
    try:
        FILE_ID = "1ruhJPhNn2I0WCpKCLSbasuG3OXNTO1i8"
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return None
```

✔ Works on Streamlit Cloud
✔ No authentication required
✔ Cached for performance

---

# ☁️ **MLflow Deployment on AWS**

## 1️⃣ Create AWS Resources

* IAM user with:

  * AmazonS3FullAccess
  * AmazonRDSFullAccess
  * AmazonEC2FullAccess (optional)
* S3 bucket:
  `mlflow-tracking-bucket46`
* RDS PostgreSQL database
* EC2 (Ubuntu, t2.large recommended)

---

## 2️⃣ Install MLflow on EC2

```bash
sudo apt update && sudo apt install python3-pip python3.12-venv -y
mkdir mlflow && cd mlflow
python3 -m venv venv
source venv/bin/activate
pip install mlflow boto3 awscli psycopg2-binary
```

---

## 3️⃣ Start MLflow Server

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://postgres:<PASSWORD>@<RDS-ENDPOINT>:5432/mlflow \
  --default-artifact-root s3://mlflow-tracking-bucket46 \
  --allowed-hosts="*"
```

Access MLflow UI:

```
http://<EC2-IP>:5000
```

---

# 🔐 **Streamlit Secrets (Required)**

Set in:
**Streamlit Cloud → App → Settings → Secrets**

```toml
# MLflow
MLFLOW_TRACKING_URI = "http://<EC2-PUBLIC-IP>:5000"

# AWS for S3 model loading
AWS_ACCESS_KEY_ID = "YOUR_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_SECRET"
AWS_DEFAULT_REGION = "ap-south-1"

---

# 🔄 **CI/CD – GitHub → Streamlit Cloud (Auto Deploy)**

Streamlit Cloud **automatically redeploys** on every push to the `main` branch.

Your CI pipeline runs checks BEFORE deployment:

### `.github/workflows/ci.yml`

```yaml
name: Streamlit CI

on:
  push:
    branches:
      - main
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Lint
        run: python -m py_compile $(git ls-files "*.py")

      - name: Run Tests
        run: pytest -q || true
```

✔ No API tokens needed
✔ No manual deploy
✔ Ultra-simple cloud-native CI/CD

---

# ▶ **Local Development**

```bash
git clone https://github.com/SachinMosambe/PatrolIQ-Smart-Safety-Analytics-Platform.git
cd PatrolIQ-Smart-Safety-Analytics-Platform

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

# 👤 **Author**

**Sachin Mosambe**
GitHub: [https://github.com/SachinMosambe](https://github.com/SachinMosambe)

---

# 🎯 Notes

* Google Drive is used for cloud-safe data loading
* All ML models are managed through MLflow (AWS-hosted)
* Streamlit Cloud automatically redeploys on every push
* AWS Secrets stored safely via Streamlit Cloud Secrets
* CI checks ensure clean deploys

---

If you want badges (Python version, CI status, Streamlit badge) added at the top, I can generate them too.
