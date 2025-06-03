# locust_tests/attackers/directory_traversal.py
import random
from locust import task

from ..base_attacker import BaseAttacker
from ..payload_generators.directory_traversal import DirectoryTraversalPayloadGenerator
from ..obfuscators import PayloadObfuscator # Needed for direct path obfuscation
from ..config import AppConfig # Needed for common query params, endpoints

class DirectoryTraversalAttacker(BaseAttacker):
    """
    Simulates a user attempting Directory Traversal (Path Traversal) attacks using dynamically generated payloads.
    """
    @task(5)
    def path_traversal_get_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        target_endpoints = ["file_download", "image_loader", "product_details"]
        path_key = random.choice(target_endpoints)
        param = random.choice(AppConfig.COMMON_QUERY_PARAMS)
        obfuscation_choice = random.choice(["single", "double", "none"])

        self._send_request(
            "get",
            path_key,
            params={param: payload},
            name_suffix="PATH_TRAVERSAL_GET_QUERY",
            obfuscation_type=obfuscation_choice
        )

        # Manual obfuscation for path segments
        if obfuscation_choice != "none":
            # --- FIX APPLIED HERE ---
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation if obfuscation_choice == "single" else PayloadObfuscator.apply_two_random_obfuscations
            obfuscated_payload_path = obfuscator_method(payload)
            # --- END FIX ---
            self.client.get(
                f"{AppConfig.ENDPOINTS[path_key]}/{obfuscated_payload_path}",
                name=f"{path_key}/[GET_PATH_TRAVERSAL_DIRECT]"
            )
        else:
            self.client.get(
                f"{AppConfig.ENDPOINTS[path_key]}/{payload}",
                name=f"{path_key}/[GET_PATH_TRAVERSAL_DIRECT_NO_OBF]"
            )


    @task(2)
    def path_traversal_post_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        obfuscation_choice = random.choice(["single", "double", "none"])
        self._send_request(
            "post",
            "feedback",
            data={"comment": "Test with path traversal", "attachment_path": payload},
            name_suffix="PATH_TRAVERSAL_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def path_traversal_put_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        obfuscation_choice = random.choice(["single", "double", "none"])

        if obfuscation_choice != "none":
            # --- FIX APPLIED HERE ---
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation if obfuscation_choice == "single" else PayloadObfuscator.apply_two_random_obfuscations
            obfuscated_payload_path = obfuscator_method(payload)
            # --- END FIX ---
            self.client.put(
                f"/upload/{obfuscated_payload_path}/test.txt",
                data=b"malicious content",
                name="/upload/[PUT_PATH_TRAVERSAL_FILENAME]"
            )
        else:
            self.client.put(
                f"/upload/{payload}/test.txt",
                data=b"malicious content",
                name="/upload/[PUT_PATH_TRAVERSAL_FILENAME_NO_OBF]"
            )


    @task(1)
    def path_traversal_delete_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        obfuscation_choice = random.choice(["single", "double", "none"])
        self._send_request(
            "delete",
            "file_download",
            params={"file": payload},
            name_suffix="PATH_TRAVERSAL_DELETE_QUERY",
            obfuscation_type=obfuscation_choice
        )