

class Parsing(Exception):
    def __init__(self, message=""):
        super().__init__(f"Parsing Error: {message}")


class CLIError(Exception):
    def __init__(self, message=""):
        super().__init__(f"CLI Error: {message}")


class FileAccessError(Exception):
    def __init__(self, message=""):
        super().__init__(f"FileAccessError: {message}")