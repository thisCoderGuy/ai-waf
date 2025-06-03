import random
import random
from locust import task

from ..base_attacker import BaseAttacker
from ..payload_generators.xss  import XSSPayloadGenerator
from ..config import AppConfig # Needed for common query params

class XSSAttacker(BaseAttacker):
    """
    Simulates a user attempting Cross-Site Scripting (XSS) attacks using dynamically generated payloads.
    """
    @task(5)
    def xss_search_fuzzing_get(self):
        payload = XSSPayloadGenerator.get_random_xss_payload() # Use the new generator
        param = random.choice(AppConfig.COMMON_QUERY_PARAMS)
        obfuscation_choice = random.choice(["single", "double", "none"])
        self._send_request(
            "get",
            "product_search",
            params={param: payload},
            name_suffix="XSS_FUZZ_GET",
            obfuscation_type=obfuscation_choice
        )

    @task(3)
    def xss_feedback_fuzzing_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload() # Use the new generator
        obfuscation_choice = random.choice(["single", "double", "none"])
        self._send_request(
            "post",
            "feedback",
            data={"comment": f"This is a test feedback with {payload}", "rating": random.randint(1, 5)},
            name_suffix="XSS_POST_COMMENT",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def xss_registration_fuzzing_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload() # Use the new generator
        password = "password123"
        obfuscation_choice = random.choice(["single", "double", "none"])

        email = f"{payload}@example.com"
        self._send_request(
            "post",
            "user_register",
            data={
                "email": email,
                "password": password,
                "passwordRepeat": password,
                "securityAnswer": "My pet's name"
            },
            name_suffix="XSS_EMAIL_POST",
            obfuscation_type=obfuscation_choice
        )

        email_clean = f"user{random.randint(1000, 9999)}@example.com"
        self._send_request(
            "post",
            "user_register",
            data={
                "email": email_clean,
                "password": password,
                "passwordRepeat": password,
                "securityAnswer": payload
            },
            name_suffix="XSS_SECURITY_ANSWER_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def xss_profile_update_put(self):
        user_id = random.randint(1, 5)
        payload = XSSPayloadGenerator.get_random_xss_payload() # Use the new generator
        obfuscation_choice = random.choice(["single", "double", "none"])
        self._send_request(
            "put",
            "user_profile",
            path_format_args={"user_id": user_id},
            data={"name": f"XSS User {payload}", "bio": "Some XSS Bio"},
            name_suffix="XSS_PROFILE_UPDATE_PUT",
            obfuscation_type=obfuscation_choice
        )