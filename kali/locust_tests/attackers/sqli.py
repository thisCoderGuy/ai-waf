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
        param = random.choice(AppConfig.COMMON_QUERY_PARAMS) # Assuming COMMON_QUERY_PARAMS is defined in AppConfig
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
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
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        # Direct path parameter injection (less common for classic SQLi, but possible for advanced cases/testing WAFs)
        # Note: Your existing product_id_fuzzing had two _send_request calls and a direct client.get.
        # I've streamlined it to use _send_request for consistency, but kept the path injection logic.
        self._send_request(
            "get",
            "product_details",
            path_format_args={"product_id": f"{product_id_base}{payload}"}, # Payload in path
            name_suffix="SQLi_FUZZ_PATH",
            obfuscation_type=obfuscation_choice
        )
        # Also include a query parameter attack for product details, as per your original intent
        self._send_request(
            "get",
            "product_details",
            path_format_args={"product_id": product_id_base}, # Legitimate path part
            params={"id": payload}, # Payload in query param 'id'
            name_suffix="SQLi_FUZZ_QUERY_ID",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def sqli_login_fuzzing(self):
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

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
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "put",
            "manage_product", 
            path_format_args={"product_id": product_id},
            data={"name": "Attacked Product", "description": f"New desc {payload}"},
            name_suffix="SQLi_PRODUCT_UPDATE_PUT",
            obfuscation_type=obfuscation_choice
        )

    

    @task(3)
    def sqli_feedback_submission(self):
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "feedbacks",
            data={
                "comment": f"SQLi Test Comment {payload}",
                "rating": random.randint(1, 5)
            },
            name_suffix="SQLi_FEEDBACK_POST",
            obfuscation_type=obfuscation_choice
        )
        self._send_request(
            "post",
            "feedbacks",
            data={
                "comment": "Legitimate comment",
                "rating": f"{random.randint(1, 5)}{payload}" # Injecting into rating (if handled as string)
            },
            name_suffix="SQLi_FEEDBACK_RATING_POST",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def sqli_basket_item_add(self):
        product_id = random.randint(1, 20)
        quantity = random.randint(1, 3)
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "post",
            "basket_items",
            data={
                "ProductId": f"{product_id}{payload}", # Injecting into ProductId
                "quantity": quantity,
                "BasketId": random.randint(1,5) # Assuming basket ID is known or found
            },
            name_suffix="SQLi_BASKET_ADD_PROD_ID",
            obfuscation_type=obfuscation_choice
        )
        self._send_request(
            "post",
            "basket_items",
            data={
                "ProductId": product_id,
                "quantity": f"{quantity}{payload}", # Injecting into quantity
                "BasketId": random.randint(1,5)
            },
            name_suffix="SQLi_BASKET_ADD_QTY",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def sqli_basket_item_update(self):
        # This requires a known basket_item_id, which is hard to get without prior interaction.
        # For a WAF dataset, we can guess common/low IDs or simulate.
        item_id = random.randint(1, 100) # Assuming some existing items
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        self._send_request(
            "put",
            "basket_item",
            path_format_args={"item_id": f"{item_id}{payload}"}, # Payload in path
            data={"quantity": random.randint(1,5)},
            name_suffix="SQLi_BASKET_UPDATE_PATH",
            obfuscation_type=obfuscation_choice
        )
        self._send_request(
            "put",
            "basket_item",
            path_format_args={"item_id": item_id},
            data={"quantity": f"{random.randint(1,5)}{payload}"}, # Payload in quantity
            name_suffix="SQLi_BASKET_UPDATE_QTY",
            obfuscation_type=obfuscation_choice
        )


    @task(3)
    def sqli_track_order_fuzzing(self):
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        # Inject into path parameter directly
        self._send_request(
            "get",
            "track_order",
            path_format_args={"order_id": f"XYZ{random.randint(1000, 9999)}{payload}"},
            name_suffix="SQLi_TRACK_ORDER_PATH",
            obfuscation_type=obfuscation_choice
        )
        # Assuming there might be a POST endpoint for tracking results
        self._send_request(
            "post",
            "track_order_result", # Check if this endpoint accepts parameters
            data={"id": f"XYZ{random.randint(1000, 9999)}{payload}"},
            name_suffix="SQLi_TRACK_ORDER_POST_ID",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def sqli_manage_address_fuzzing(self):
        address_id = random.randint(1, 10) # Assuming some existing addresses
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        # Inject into address ID in path (for update/delete)
        self._send_request(
            "get", # Or put/delete
            "address",
            path_format_args={"address_id": f"{address_id}{payload}"},
            name_suffix="SQLi_ADDRESS_ID_PATH",
            obfuscation_type=obfuscation_choice
        )

        # Inject into address creation fields
        self._send_request(
            "post",
            "addresses",
            data={
                "fullName": f"SQLi User {payload}",
                "country": "USA",
                "zipCode": "12345",
                "streetAddress": "Test Street",
                "city": "Test City",
                "state": "TS"
            },
            name_suffix="SQLi_ADDRESS_CREATE",
            obfuscation_type=obfuscation_choice
        )

    @task(2)
    def sqli_manage_card_fuzzing(self):
        card_id = random.randint(1, 10) # Assuming some existing cards
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        # Inject into card ID in path (for update/delete)
        self._send_request(
            "get", # Or put/delete
            "card",
            path_format_args={"card_id": f"{card_id}{payload}"},
            name_suffix="SQLi_CARD_ID_PATH",
            obfuscation_type=obfuscation_choice
        )

        # Inject into card creation fields
        self._send_request(
            "post",
            "cards",
            data={
                "fullName": f"SQLi Card {payload}",
                "cardNum": f"123456789012345{random.randint(0,9)}",
                "expMonth": random.randint(1,12),
                "expYear": random.randint(2025,2030)
            },
            name_suffix="SQLi_CARD_CREATE",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def sqli_manage_privacy_request_fuzzing(self):
        privacy_id = random.randint(1, 5)
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        # Inject into ID in path
        self._send_request(
            "get",
            "privacy_request",
            path_format_args={"privacy_request_id": f"{privacy_id}{payload}"},
            name_suffix="SQLi_PRIVACY_REQ_ID_PATH",
            obfuscation_type=obfuscation_choice
        )
        # Inject into creation fields
        self._send_request(
            "post",
            "privacy_requests",
            data={
                "subject": f"Data Access Request {payload}",
                "message": "Please provide my data."
            },
            name_suffix="SQLi_PRIVACY_REQ_CREATE",
            obfuscation_type=obfuscation_choice
        )

    @task(1)
    def sqli_memory_details_fuzzing(self):
        memory_id = random.randint(1, 20)
        payload = SQLiPayloadGenerator.get_random_sqli_payload()
        obfuscation_choice = random.choice(BaseAttacker.OBFUSCATION_CHOICES)

        # Inject into path parameter for memory details
        self._send_request(
            "get",
            "memory_detail",
            path_format_args={"id": f"{memory_id}{payload}"},
            name_suffix="SQLi_MEMORY_ID_PATH",
            obfuscation_type=obfuscation_choice
        )
        # Inject into memory creation fields
        self._send_request(
            "post",
            "memories",
            data={
                "caption": f"My memory {payload}",
                "image": "some_base64_encoded_image_string"
            },
            name_suffix="SQLi_MEMORY_CREATE",
            obfuscation_type=obfuscation_choice
        )

  



