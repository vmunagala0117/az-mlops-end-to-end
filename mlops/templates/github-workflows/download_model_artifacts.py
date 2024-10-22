import argparse
import mlflow
import os
from mlflow.tracking import MlflowClient

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Download the MLflow model artifacts for a specific version.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_version", type=str, required=True, help="Version of the model to download")
    parser.add_argument("--workspace_uri", type=str, required=True, help="MLflow tracking URI for the Azure ML workspace")
    parser.add_argument("--download_path", type=str, required=True, help="Local path to download the model artifacts")
    return parser.parse_args()

def download_model_artifacts(model_name, model_version, download_path):
    '''Download the artifacts of the specified model version'''
    client = MlflowClient()

    # Build the model URI
    model_uri = f"models:/{model_name}/{model_version}"
    
    # Download the artifacts
    try:
        local_path = client.download_artifacts(run_id=f"runs:/{model_name}/{model_version}", path="", dst_path=download_path)
        print(f"Model artifacts downloaded to: {local_path}")
    except Exception as e:
        print(f"Error downloading model artifacts: {e}")
        exit(1)

def main(args):
    # Set the MLflow tracking URI to the Azure ML workspace
    os.environ["MLFLOW_TRACKING_URI"] = args.workspace_uri
    print(f"MLflow tracking URI set to: {args.workspace_uri}")

    # Download the model artifacts
    download_model_artifacts(args.model_name, args.model_version, args.download_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
