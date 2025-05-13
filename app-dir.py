import os

structure = {
    "triage-pose": {
        "app": {
            "__init__.py": "",
            "main.py": "# FastAPI application\n",
            "config.py": "# Application configuration\n",
            "routers": {
                "__init__.py": "",
                "assessment.py": "# Assessment API endpoints\n",
                "exercise.py": "# Exercise guidance endpoints\n",
                "utils.py": "# Utility endpoints\n",
            },
            "services": {
                "__init__.py": "",
                "video_service.py": "# Video processing (from physiotrack-api)\n",
                "analysis_service.py": "# ROM analysis (from physiotrack-api)\n",
                "streaming_service.py": "# Real-time streaming\n",
                "storage_service.py": "# Results storage\n",
            },
            "processors": {
                "__init__.py": "",
                "pose_processor.py": "# Pose processing workflow\n",
                "rom_processor.py": "# ROM processing workflow\n",
                "filtering.py": "# Data filtering (from physiotrack/filter.py)\n",
            },
            "visualization": {
                "__init__.py": "",
                "angle_plots.py": "# Angle visualization\n",
                "rom_plots.py": "# ROM visualization\n",
                "frame_utils.py": "# Frame annotation and drawing\n",
            },
            "models": {
                "__init__.py": "",
                "request.py": "# API request models\n",
                "response.py": "# API response models\n",
                "data.py": "# Internal data models\n",
            },
            "io": {
                "__init__.py": "",
                "video.py": "# Video handling functions\n",
                "trc.py": "# TRC file handling\n",
                "mot.py": "# MOT file handling\n",
                "json_utils.py": "# JSON utilities\n",
            }
        },
        "static": {},
        "templates": {},
        "tests": {},
        "scripts": {},
        ".env": "",
        "docker-compose.yml": "# Docker compose configuration\n",
        "Dockerfile": "# Dockerfile for the application\n",
        "requirements.txt": "# Dependencies including physiotrack\n"
    }
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)

# Create the structure
create_structure(".", structure)
