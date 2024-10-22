import argparse
import mlflow
from mlflow.exceptions import RestException

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the registered model in MLflow',
        required=True)
    parser.add_argument(
        '--version',
        type=str,
        help='Version of the model to promote. If not provided, the latest version will be used.',
        default=None)
    parser.add_argument(
        '--alias',
        type=str,
        help='Alias to assign to the model version (e.g., production_model)',
        required=True)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def get_latest_model_version(model_name):
    '''Fetches the latest version of the model'''
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        # Sort versions by their version number in descending order and return the latest one
        latest_version = max(versions, key=lambda v: int(v.version))
        print(f"Latest version for model '{model_name}' is: {latest_version.version}")
        return latest_version.version
    except RestException as e:
        print(f"Error fetching latest version for model '{model_name}': {str(e)}")
        return None

def promote_model(model_name, version, alias):
    '''Promotes the model by assigning an alias'''

    try:
        # Set the alias for the model version
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version
        )

        print(f"Model '{model_name}' version '{version}' assigned alias '{alias}' successfully.")

    except RestException as e:
        print(f"Error promoting model: {str(e)}")

def main(args):
    '''Main function to promote the model'''

    # If version is not provided, fetch the latest version
    if not args.version:
        print("Version not provided. Fetching the latest version...")
        latest_version = get_latest_model_version(args.model_name)
        if not latest_version:
            print("No versions found for the model. Exiting.")
            return
        args.version = latest_version

    print(f"Promoting model '{args.model_name}' version '{args.version}' with alias '{args.alias}'")

    promote_model(args.model_name, args.version, args.alias)

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Promote the model
    main(args)
