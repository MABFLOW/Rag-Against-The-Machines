

class ParsingError(Exception):
    """Raised when data fails to parse into an expected schema or format."""

    def __init__(self, message: str = "") -> None:
        """Initializes the error.

        Args:
            message: Description of what failed to parse.
        """
        super().__init__(f"Parsing Error: {message}")


class CLIError(Exception):
    """Raised when a command-line argument is missing or invalid."""

    def __init__(self, message: str = "") -> None:
        """Initializes the error.

        Args:
            message: Description of the invalid CLI usage.
        """
        super().__init__(f"CLI Error: {message}")


class FileAccessError(Exception):
    """Raised when a file cannot be accessed on disk."""

    def __init__(self, message: str = "") -> None:
        """Initializes the error.

        Args:
            message: Description of the file access failure.
        """
        super().__init__(f"FileAccessError: {message}")
