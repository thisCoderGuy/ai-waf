import random
from locust import TaskSet

# Import from other files in the same package
from .config import AppConfig
from .obfuscators import PayloadObfuscator

# Import specific generators (or the whole subpackage if you prefer, but explicit is clearer)
from .payload_generators.sqli import SQLiPayloadGenerator
from .payload_generators.xss import XSSPayloadGenerator
from .payload_generators.directory_traversal import DirectoryTraversalPayloadGenerator


class BaseAttacker(TaskSet):
    """
    A base class for all attacker TaskSets, providing common utility methods for different HTTP methods.
    """
    def _send_request(self, method: str, path_key: str, data: dict = None, params: dict = None,
                      name_suffix: str = "", path_format_args: dict = None, headers: dict = None,
                      obfuscation_type: str = "single"):
        """
        Helper to send various HTTP requests with obfuscated payloads.
        method: HTTP method (e.g., 'get', 'post', 'put', 'delete', 'patch')
        path_key: Key from AppConfig.ENDPOINTS
        data: JSON or form data for POST/PUT/PATCH
        params: Query parameters for GET/DELETE/PUT/PATCH
        name_suffix: Suffix for Locust request name for better reporting
        path_format_args: Dictionary for formatting path (e.g., product_id=123)
        headers: Additional headers for the request
        obfuscation_type: 'single', 'double', or 'none' for no obfuscation.
        """
        path = AppConfig.ENDPOINTS[path_key]
        if path_format_args:
            path = path.format(**path_format_args)

        full_name = f"{path_key}/[{method.upper()}_{name_suffix}]"

        request_kwargs = {
            "name": full_name,
            "headers": headers if headers else {},
        }

        apply_obfuscation = obfuscation_type != "none"

        if params and apply_obfuscation:
            obfuscated_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    if obfuscation_type == "double":
                        obfuscated_params[k] = PayloadObfuscator.apply_two_random_obfuscations(v)
                    else:
                        obfuscated_params[k] = PayloadObfuscator.apply_random_obfuscation(v)
                else:
                    obfuscated_params[k] = v
            request_kwargs["params"] = obfuscated_params
        elif params:
            request_kwargs["params"] = params

        if data and apply_obfuscation:
            obfuscated_data = {}
            for k, v in data.items():
                if isinstance(v, str):
                    if obfuscation_type == "double":
                        obfuscated_data[k] = PayloadObfuscator.apply_two_random_obfuscations(v)
                    else:
                        obfuscated_data[k] = PayloadObfuscator.apply_random_obfuscation(v)
                else:
                    obfuscated_data[k] = v
            request_kwargs["json"] = obfuscated_data
        elif data:
            request_kwargs["json"] = data

        getattr(self.client, method.lower())(path, **request_kwargs)
