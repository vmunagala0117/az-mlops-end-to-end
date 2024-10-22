import argparse
import mlflow
import os
from mlflow.tracking import MlflowClient

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Query for the Production model version.")
    parser.add_argument("--workspace_uri", type=str, required=True, help="MLFLOW tracking URI")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to query")
    return parser.parse_args()

def get_production_model_version(model_name):
    '''Fetch the model version marked as Production'''
    client = MlflowClient()

    # Fetch all versions of the specified model
    versions = client.search_model_versions(f"name='{model_name}'")

    # Find and return the version marked as Production
    for version in versions:
        if version.current_stage == "Production":
            print(f"Model '{model_name}' version '{version.version}' is in Production.")
            return version.version
    
    print(f"No version of model '{model_name}' is currently in Production.")
    return None

def main(args):
    # Set the MLflow tracking URI to the Azure ML workspace
    #tracking_uri = f"https://{args.workspace_name}.api.azureml.ms/mlflow/v1.0/subscriptions/{args.subscription_id}/resourceGroups/{args.resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{args.workspace_name}"
    os.environ["MLFLOW_TRACKING_URI"] = args.workspace_uri
    print(f"MLflow tracking URI set to: {args.workspace_uri}")

    # Query for the production model version
    version = get_production_model_version(args.model_name)
    if version:
        print(f"Production model version: {version}")
    else:
        print("No production model version found.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
