import os
from pathlib import Path
import logging

#logging string
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s')

project_name = input("Multimodal Search Engine")

# Define the file structure
list_of_files = [
    # Root level
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "README.md",
    "docker/Dockerfile",

    # src
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/download_data.py",
    f"src/{project_name}/data/data_utils.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/model_architectures.py",
    f"src/{project_name}/models/model_training.py",
    f"src/{project_name}/models/model_evaluation.py",
    f"src/{project_name}/models/model_inference.py",
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/app.py",
    f"src/{project_name}/api/routes.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/config.py",
    f"src/{project_name}/utils/logger.py",

    # tests
    "tests/__init__.py",
    "tests/test_data.py",
    "tests/test_models.py",
    "tests/test_api.py",
]




for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info("Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            if 'common.py' in filepath.name:
                f.write(common_template_text)
            if 'setup.py' in filepath.name:
                f.write(setup_template_text) 
            else:
                f.write('')
        logging.info("Creating file: {filename} in directory: {filedir}")

    else:
        logging.info("{filename} is already exists")
