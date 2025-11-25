# save as promote_models.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://13.233.98.214:5000")
client = MlflowClient()

print("=" * 60)
print("Promoting Models to Production")
print("=" * 60)

models_to_promote = [
    "PatrolIQ_Best_Model",
    "PatrolIQ_Temporal_Clustering_Model", 
    "PatrolIQ_Dimensionality_Reduction_Model"
]

for model_name in models_to_promote:
    try:
        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"\n⚠️  {model_name}: No versions found")
            continue
        
        # Get version 1 (the one we saw in diagnostics)
        version_to_promote = "1"
        
        print(f"\n📦 {model_name}")
        print(f"   Current stage: None")
        print(f"   Promoting version {version_to_promote} to Production...")
        
        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version_to_promote,
            stage="Production",
            archive_existing_versions=False
        )
        
        print(f"   ✅ Successfully promoted to Production!")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n" + "=" * 60)
print("Verifying promotion...")
print("=" * 60)

for model_name in models_to_promote:
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            stage_icon = "🟢" if v.current_stage == "Production" else "⚪"
            print(f"{stage_icon} {model_name} v{v.version}: {v.current_stage}")
    except Exception as e:
        print(f"❌ {model_name}: {e}")

print("\n✅ Done! Now try loading your models in the Streamlit app.")