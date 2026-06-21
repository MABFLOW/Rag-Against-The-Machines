

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

    
