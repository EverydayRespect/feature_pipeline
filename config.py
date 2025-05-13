import yaml

def read_config(path="config.yaml"):
    """
    Reads the configuration file and returns the configuration as a dictionary.
    
    Args:
        path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration settings.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config