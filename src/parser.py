from .exceptions import CLIError, ParsingError, FileAccessError
from pathlib import Path
import json
from typing import Any


class Parser:
    """Provides file I/O and CLI argument validation helpers."""

    def read_from_file(self, file_path: Path | str) -> str:
        """Reads and returns the text content of a file.

        Args:
            file_path: Path to the file to read.

        Returns:
            The file's decoded text content.

        Raises:
            PermissionError: If the file cannot be accessed due to
                permissions.
            FileNotFoundError: If the file does not exist.
            Exception: If any other error occurs while reading the file.
        """
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

    def validate_number(self, var: Any, name: str) -> None:
        """Validates that a value is a positive integer.

        Args:
            var: The value to validate.
            name: Human-readable name of the value, used in error messages.

        Raises:
            CLIError: If ``var`` is not a positive integer.
        """
        if isinstance(var, bool):
            raise CLIError("--k must be an integer (e.g. --k=5).")
        if not isinstance(var, int):
            raise CLIError(f"{name} should be a valid integer.")
        if var <= 0:
            raise CLIError(f"{name} must be greater than 0.")

    def validate_argument(self, var: Any, name: str) -> None:
        """Validates that a value is a string.

        Args:
            var: The value to validate.
            name: Human-readable name of the value, used in error messages.

        Raises:
            CLIError: If ``var`` is not a string.
        """
        if not isinstance(var, str):
            raise CLIError(f"{name} must be a valid str.")

    def validate_file(self, path: Path | str, name: str) -> None:
        """Validates that a path points to a non-empty, readable file.

        Args:
            path: The path to validate.
            name: Human-readable name of the path, used in error messages.

        Raises:
            CLIError: If ``path`` does not point to a file.
            ParsingError: If the file is empty.
        """
        file = Path(str(path))
        if not file.is_file():
            raise CLIError(f"'{name}' Path must be valid.")
        content = self.read_from_file(file)
        if not content:
            raise ParsingError(f"'{name}' is empty.")

    def validate_dir(self, path: Path | str, name: str) -> Path:
        """Validates a directory path, creating it if it does not exist.

        Args:
            path: The directory path to validate.
            name: Human-readable name of the path, used in error messages.

        Returns:
            The validated directory path.

        Raises:
            CLIError: If ``path`` does not resolve to a valid directory.
        """
        try:
            dir = Path(str(path))
            dir.mkdir(parents=True, exist_ok=True)
            return dir
        except Exception:
            pass
        if not dir.is_dir():
            raise CLIError(f"'{name}' Path must be valid.")
        return dir

    def load_data(self, path: Path | str) -> Any:
        """Loads and parses JSON data from a file.

        Args:
            path: Path to the JSON file to load.

        Returns:
            The parsed JSON data.

        Raises:
            FileAccessError: If the file cannot be accessed due to
                permissions.
            ParsingError: If the file does not contain valid JSON.
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except PermissionError:
            raise FileAccessError(
                f"Permission denied while accessing '{path}'.")
        except json.JSONDecodeError:
            raise ParsingError(f"Data in '{path}' must be valid JSON.")

    def dump_to_dir(self, path: Path | str, data: Any) -> None:
        """Serializes a pydantic model to a JSON file.

        Args:
            path: Destination file path.
            data: A pydantic model instance with a ``model_dump`` method.
        """
        try:
            with open(path, 'w') as f:
                json.dump(data.model_dump(), f, indent=2)
        except PermissionError:
            raise FileAccessError(f"{path} Permission Denied.")
        except Exception:
            raise FileAccessError(f"{path} Something went wrong while Accessing...")

    
        
