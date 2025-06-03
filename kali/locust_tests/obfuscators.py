import random
import urllib.parse
import binascii

class PayloadObfuscator:
    """
    Provides various methods to obfuscate attack payloads.
    """
    @staticmethod
    def url_encode(payload: str) -> str:
        """Applies standard URL encoding."""
        return urllib.parse.quote_plus(payload)

    @staticmethod
    def double_url_encode(payload: str) -> str:
        """Applies URL encoding twice."""
        return urllib.parse.quote_plus(urllib.parse.quote_plus(payload))

    @staticmethod
    def html_entity_encode(payload: str) -> str:
        """Converts characters in the payload to HTML entities."""
        return "".join(f"&#x{ord(c):x};" for c in payload)

    @staticmethod
    def case_vary(payload: str) -> str:
        """Randomly changes the case of characters in the payload."""
        return "".join(random.choice([c.lower(), c.upper()]) if c.isalpha() else c for c in payload)

    @staticmethod
    def hex_encode(payload: str) -> str:
        """Converts the payload to hex encoded format (e.g., %HH)."""
        return "".join(f"%{binascii.hexlify(c.encode()).decode()}" for c in payload)

    @staticmethod
    def _get_obfuscators_list():
        """Helper to get a list of all static obfuscation methods."""
        return [
            PayloadObfuscator.url_encode,
            PayloadObfuscator.double_url_encode,
            PayloadObfuscator.html_entity_encode,
            PayloadObfuscator.case_vary,
            PayloadObfuscator.hex_encode,
        ]

    @staticmethod
    def apply_random_obfuscation(payload: str) -> str:
        """Applies a single random obfuscation technique to the payload."""
        obfuscators = PayloadObfuscator._get_obfuscators_list()
        chosen_obfuscator = random.choice(obfuscators)
        return chosen_obfuscator(payload)

    @staticmethod
    def apply_two_random_obfuscations(payload: str) -> str:
        """Applies two distinct random obfuscation techniques sequentially to the payload."""
        obfuscators = PayloadObfuscator._get_obfuscators_list()

        if len(obfuscators) < 2:
            return PayloadObfuscator.apply_random_obfuscation(payload)

        obf1, obf2 = random.sample(obfuscators, 2)

        temp_payload = obf1(payload)
        final_payload = obf2(temp_payload)
        return final_payload