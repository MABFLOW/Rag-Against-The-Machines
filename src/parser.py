from .exceptions import *
from pathlib import Path
import json
from .models import *

class Parser:

    def read_from_file(self, file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                content = f.read()
            return content
        except PermissionError:
            raise PermissionError(f"Permission Denied: ({file_path})")
        except FileNotFoundError:
            raise FileNotFoundError(f"File Not Found: ({file_path})")
        except Exception:
            raise Exception(f"Something Went Wrong: {file_path}")

    def validate_number(self, var, name):
        if isinstance(var, bool):
            raise CLIError("--k must be an integer (e.g. --k=5).")
        if not isinstance(var, int):
            raise CLIError(f"{name} should be a valid integer.")
        if var <= 0:
            raise CLIError(f"{name} must be greater than 0.")

    def validate_argument(self, var, name):
        if isinstance(var, bool):
            raise CLIError(f"{name} must be a valid str.\nNote: True/False are treated as booleans not str. if you dont want that for ex use 'True'")

    def validate_file(self, path, name):
        file = Path(str(path))
        if not file.is_file():
            raise CLIError(f"'{name}' Path must be valid.")

    def validate_dir(self, path, name):
        dir = Path(str(path))
        dir.mkdir(parents=True, exist_ok=True)
        return dir
        if not dir.is_dir():
            raise CLIError(f"'{name}' Path must be valid.")

            
    def load_data(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except PermissionError:
            raise FileAccessError(f"Permission denied while accessing '{path}'.")
        

    def load_unanswered_question(self, dataset):
        data = []
        rag_questions = dataset.get('rag_questions')

        for item in rag_questions:
            data.append(UnansweredQuestion(
                question_id=item.get("question_id"), 
                question=item.get("question")
        ))

        return data


    def dump_to_dir(self, path, data):
        with open(path, 'w') as f:
            json.dump(data.model_dump(), f, indent=2)

    
