# locust_tests/attackers/sqli.py
import random
from locust import task

from ..base_attacker import BaseAttacker
from ..payload_generators.sqli import SQLiPayloadGenerator
from ..obfuscators import PayloadObfuscator # Needed for direct path obfuscation
from ..config import AppConfig # Needed for common query params, endpoints

class SQLiAttacker(BaseAttacker):
    """
    Simulates a user attempting SQL Injection attacks using dynamically generated payloads.
    """
    @task(5)
    def sqli_search_fuzzing_get(self):
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        param = random.choice(AppConfig.COMMON_QUERY_PARAMS)
        obfuscation_choice = random.choice(["single", "double", "none"])
        self._send_request(
            "get",
            "product_search",
            params={param: payload},
            name_suffix="SQLi_FUZZ_GET",
            obfuscation_type=obfuscation_choice
        )

    @task(3)
    def sqli_product_id_fuzzing(self):
        product_id_base = random.randint(1, 10)
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(["single", "double", "none"])

        if obfuscation_choice != "none":
            # --- FIX APPLIED HERE ---
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation if obfuscation_choice == "single" else PayloadObfuscator.apply_two_random_obfuscations
            obfuscated_payload_path = obfuscator_method(payload)
            # --- END FIX ---

            self.client.get(
                AppConfig.ENDPOINTS["product_details"].format(product_id=f"{product_id_base}{obfuscated_payload_path}"),
                name=f"{AppConfig.ENDPOINTS['product_details']}/[GET_SQLi_FUZZ_PATH]"
            )
        else:
            self.client.get(
                AppConfig.ENDPOINTS["product_details"].format(product_id=f"{product_id_base}{payload}"),
                name=f"{AppConfig.ENDPOINTS['product_details']}/[GET_SQLi_FUZZ_PATH_NO_OBF]"
            )

        self._send_request(
            "get",
            "product_details",
            path_format_args={"product_id": product_id_base},
            params={"id": payload},
            name_suffix="SQLi_FUZZ_QUERY_ID",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def sqli_login_fuzzing(self):
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(["single", "double", "none"])

        self._send_request(
            "post",
            "user_login",
            data={"email": f"admin{payload}@example.com", "password": "password"},
            name_suffix="SQLi_EMAIL_POST",
            obfuscation_type=obfuscation_choice
        )

        self._send_request(
            "post",
            "user_login",
            data={"email": "admin@example.com", "password": f"password{payload}"},
            name_suffix="SQLi_PASSWORD_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def sqli_update_product_put(self):
        product_id = random.randint(1, 10)
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(["single", "double", "none"])
        self._send_request(
            "put",
            "update_product",
            path_format_args={"product_id": product_id},
            data={"name": "Attacked Product", "description": f"New desc {payload}"},
            name_suffix="SQLi_PRODUCT_UPDATE_PUT",
            obfuscation_type=obfuscation_choice
        )