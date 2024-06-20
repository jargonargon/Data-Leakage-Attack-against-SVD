import yaml

def load_config(config_path):
    """
    Load settings from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config["DataConfig"], config["AdversaryKnowledge"], config["DifferentialEvolutionConfig"]

def load_config_forEvaluation(config_path):
    """
    Load settings from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config["useAttackConfig"], config["DataConfig"], config["AdversaryKnowledge"]