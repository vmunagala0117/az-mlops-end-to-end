import argparse
import json
import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Query for the Production model version.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to query")
    parser.add_argument("--resource_group", type=str, required=True, help="Resource group for Azure ML workspace")
    parser.add_argument("--workspace_name", type=str, required=True, help="Azure ML workspace name")
    parser.add_argument("--download_path", type=str, required=True, help="Local path to download the model artifacts")
    return parser.parse_args()

def get_azure_credential():
    '''Retrieve Azure credentials from the environment variable'''
    print("printing az creds...")
    azure_credentials = os.environ.get('AZURE_CREDENTIALS')
    creds = json.loads(azure_credentials)
    tenant_id = creds["tenantId"]
    client_id = creds["clientId"]
    client_secret = creds["clientSecret"]
    subscription_id = creds.get("subscriptionId", None)

    # Set up the ClientSecretCredential for Azure authentication
    credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
    return credential, subscription_id

def download_model_artifacts(client, model_name, model_version, download_path):
    '''Download the artifacts of the specified model version'''
    
    client.models.download(name=model_name, version=model_version, download_path=download_path)
    
    print(f"Model artifacts for '{model_name}' version '{model_version}' downloaded to: {download_path}")
        
def get_production_model_version(client, model_name):
    '''Fetch the model version marked as Production'''
    versions = client.models.list(name=model_name)
    for version in versions:
        if version.stage == "Production":
            print(f"Model '{model_name}' version '{version.version}' is in '{version.stage}'.")
            return version.version
    
    print(f"No version of model '{model_name}' is currently in Production.")
    return None

def main(args):
    # Get credentials and subscription ID
    credential, subscription_id = get_azure_credential()
    
    # Set up the Azure ML client
    ml_client = MLClient(
        credential,
        subscription_id=subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )

    # Query for the production model version using the Azure ML SDK
    version = get_production_model_version(ml_client, args.model_name)
    if version:
        print(f"Production model version: {version}")
    else:
        print("No production model version found.")
    
    download_model_artifacts(ml_client, args.model_name, version, args.download_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
