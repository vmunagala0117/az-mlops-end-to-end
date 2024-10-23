import argparse
import json
import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from mlflow.tracking import MlflowClient

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Query for the Production model version.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to query")
    parser.add_argument("--resource_group", type=str, required=True, help="Resource group for Azure ML workspace")
    parser.add_argument("--workspace_name", type=str, required=True, help="Azure ML workspace name")
    return parser.parse_args()

def get_azure_credential():
    '''Retrieve Azure credentials from the environment variable'''
    try:
        print("printing az creds...")
        print(os.getenv("AZURE_CREDENTIALS"))        
        azure_credentials = json.loads(os.getenv("AZURE_CREDENTIALS"))
        tenant_id = azure_credentials["tenantId"]
        client_id = azure_credentials["clientId"]
        client_secret = azure_credentials["clientSecret"]
        subscription_id = azure_credentials.get("subscriptionId", None)

        # Set up the ClientSecretCredential for Azure authentication
        credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
        return credential, subscription_id
    except json.JSONDecodeError as e:
        print(f"Error parsing AZURE_CREDENTIALS: {e}")
        exit(1)

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
    # Set the Azure ML tracking URI using the Azure ML SDK
    credential, subscription_id = get_azure_credential()
    ml_client = MLClient(
        credential,
        subscription_id=subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )

    # Retrieve and set the MLFLOW_TRACKING_URI based on the workspace information
    tracking_uri = ml_client.workspaces.get(args.workspace_name).mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    print(f"MLflow tracking URI set to: {tracking_uri}")

    # Query for the production model version
    version = get_production_model_version(args.model_name)
    if version:
        print(f"Production model version: {version}")
    else:
        print("No production model version found.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
