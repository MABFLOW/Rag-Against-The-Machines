

class Parser:

    def __init__(self, file_path):
        self.file_path = file_path
        self.content = ""

    def read_from_file(self):
        try:
            with open(self.file_path, 'r') as f:
                self.content = f.read()
        except PermissionError:
            raise PermissionError(f"Permission Denied: ({self.file_path})")
        except FileNotFoundError:
            raise FileNotFoundError(f"File Not Found: ({self.file_path})")
        except Exception:
            raise Exception(f"Something Went Wrong: {self.file_path}")

    
