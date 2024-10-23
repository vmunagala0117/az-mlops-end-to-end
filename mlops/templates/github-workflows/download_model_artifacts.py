import argparse
import json
import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Download the Azure ML model artifacts for a specific version.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_version", type=str, required=True, help="Version of the model to download")
    parser.add_argument("--resource_group", type=str, required=True, help="Resource group for Azure ML workspace")
    parser.add_argument("--workspace_name", type=str, required=True, help="Azure ML workspace name")
    parser.add_argument("--download_path", type=str, required=True, help="Local path to download the model artifacts")
    return parser.parse_args()

def get_azure_credential():
    '''Retrieve Azure credentials from the environment variable'''
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
    try:
        # Get the model by name and version
        model = client.models.get(name=model_name, version=model_version)

        # Download the model artifacts to the specified path
        client.models.download(name=model_name, version=model_version, download_path=download_path)
        print(f"Model artifacts for '{model_name}' version '{model_version}' downloaded to: {download_path}")
    except Exception as e:
        print(f"Error downloading model artifacts: {e}")
        exit(1)

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

    # Download the model artifacts using Azure ML SDK
    download_model_artifacts(ml_client, args.model_name, args.model_version, args.download_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
