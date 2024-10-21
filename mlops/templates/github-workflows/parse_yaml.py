import yaml
import sys

def parse_yaml(file_path):
    # Open the YAML file and parse it
    with open(file_path, 'r') as file:
        env_vars = yaml.safe_load(file)
    
    # Extract the 'variables' section
    variables = env_vars.get('variables', {})

    # Iterate over each variable and set it as a GitHub Actions output
    for key, value in variables.items():
        # Print the output in the format that GitHub Actions understands
        print(f"::set-output name={key}::{value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_yaml.py <path_to_yaml_file>")
        sys.exit(1)
    
    yaml_file_path = sys.argv[1]
    parse_yaml(yaml_file_path)
