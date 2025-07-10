import random
from locust import task

from ..base_attacker import BaseAttacker
from ..config import AppConfig # Needed for CSRF headers, endpoints

#TODO Add obfuscation


class CSRFAttacker(BaseAttacker):
    """
    Simulates Cross-Site Request Forgery (CSRF) attacks by sending requests
    to sensitive endpoints without a valid CSRF token, or with a dummy one.
    This assumes the application expects a CSRF token for these actions.
    """

    def _send_csrf_request(self, method: str, path_key: str, data: dict = None, params: dict = None,
                           name_suffix: str = "", path_format_args: dict = None, include_invalid_token: bool = False):
        """
        Helper method specifically for CSRF attacks.
        It calls the base _send_request with obfuscation_type="none"
        and optionally adds invalid CSRF headers.
        """
        headers = None
        if include_invalid_token:
            headers = AppConfig.CSRF_HEADERS # Use the predefined invalid token headers

        # Call the BaseAttacker's _send_request, explicitly setting obfuscation_type="none"
        super()._send_request( 
            method=method,
            path_key=path_key,
            data=data,
            params=params,
            name_suffix=name_suffix,
            path_format_args=path_format_args,
            headers=headers,
            obfuscation_type="none" # Key for CSRF: no payload obfuscation
        )

   

    @task(3)
    def csrf_add_to_cart_post(self):
        item_id = random.randint(1, 50)
        quantity = random.randint(1, 5)
        self._send_csrf_request(
            "post",
            "add_to_cart",
            data={"itemId": item_id, "quantity": quantity},
            name_suffix="ADD_TO_CART_NO_TOKEN_POST"
        )

    @task(2)
    def csrf_password_reset_attempt_post(self):
        new_password = "new_password_123"
        self._send_csrf_request(
            "post",
            "password_reset",
            data={
                "email": f"user{random.randint(100,999)}@example.com",
                "answer": "My pet's name",
                "new": new_password,
                "repeat": new_password
            },
            name_suffix="PASSWORD_RESET_NO_TOKEN_POST"
        )

    @task(1)
    def csrf_admin_action_attempt_post(self):
        target_user_id = random.randint(1, 10)
        self._send_csrf_request(
            "put",
            "admin_panel",
            path_format_args={"user_id": target_user_id},
            data={"role": "customer"},
            name_suffix="ADMIN_DEMOTE_USER_NO_TOKEN_PUT"
        )
        self._send_csrf_request(
            "delete",
            "admin_panel",
            path_format_args={"user_id": target_user_id},
            name_suffix="ADMIN_DELETE_USER_NO_TOKEN_DELETE"
        )


    @task(1)
    def csrf_with_invalid_token_post(self):
        item_id = random.randint(1, 50)
        quantity = random.randint(1, 5)
        self._send_csrf_request(
            "post",
            "add_to_cart",
            data={"itemId": item_id, "quantity": quantity},
            name_suffix="ADD_TO_CART_INVALID_TOKEN_POST",
            include_invalid_token=True # Flag to include the invalid token
        )

    @task(1)
    def csrf_update_product_put(self):
        product_id = random.randint(1, 10)
        self._send_csrf_request(
            "put",
            "update_product",
            path_format_args={"product_id": product_id},
            data={"name": "CSRF Updated Product", "price": round(random.uniform(10.0, 100.0), 2)},
            name_suffix="UPDATE_PRODUCT_NO_TOKEN_PUT"
        )

    @task(1)
    def csrf_delete_product_delete(self):
        product_id = random.randint(1, 10)
        self._send_csrf_request(
            "delete",
            "delete_product",
            path_format_args={"product_id": product_id},
            name_suffix="DELETE_PRODUCT_NO_TOKEN_DELETE"
        )

    # --- NEW CSRF TASKS (UPDATED to use _send_csrf_request) ---

    @task(2)
    def csrf_user_profile_update_put(self):
        user_id = random.randint(1, 5)
        self._send_csrf_request(
            "put",
            "user_profile_update",
            path_format_args={"user_id": user_id},
            data={"email": f"csrf_victim_{random.randint(100,999)}@example.com", "name": "CSRF Victim"},
            name_suffix="USER_PROFILE_UPDATE_NO_TOKEN_PUT"
        )

    @task(2)
    def csrf_place_order_post(self):
        self._send_csrf_request(
            "post",
            "place_order",
            data={
                "customerId": random.randint(1, 10),
                "deliveryAddressId": random.randint(1, 10),
                "paymentId": random.randint(1, 10)
            },
            name_suffix="PLACE_ORDER_NO_TOKEN_POST"
        )

    @task(2)
    def csrf_add_address_post(self):
        self._send_csrf_request(
            "post",
            "addresses",
            data={
                "fullName": "CSRF Attacker Address",
                "country": "US",
                "zipCode": "99999",
                "streetAddress": "123 CSRF Lane",
                "city": "Testville",
                "state": "TS"
            },
            name_suffix="ADD_ADDRESS_NO_TOKEN_POST"
        )

    @task(1)
    def csrf_delete_address_delete(self):
        address_id = random.randint(1, 10)
        self._send_csrf_request(
            "delete",
            "address",
            path_format_args={"address_id": address_id},
            name_suffix="DELETE_ADDRESS_NO_TOKEN_DELETE"
        )

    @task(1)
    def csrf_add_card_post(self):
        self._send_csrf_request(
            "post",
            "cards",
            data={
                "fullName": "CSRF Attacker Card",
                "cardNum": f"111122223333{random.randint(1000, 9999)}",
                "expMonth": random.randint(1, 12),
                "expYear": random.randint(2025, 2030)
            },
            name_suffix="ADD_CARD_NO_TOKEN_POST"
        )

    @task(1)
    def csrf_delete_card_delete(self):
        card_id = random.randint(1, 10)
        self._send_csrf_request(
            "delete",
            "card",
            path_format_args={"card_id": card_id},
            name_suffix="DELETE_CARD_NO_TOKEN_DELETE"
        )

    @task(1)
    def csrf_change_security_answer_post(self):
        self._send_csrf_request(
            "post",
            "change_security_answer",
            data={"question": "What is my pet's name?", "answer": "csrf_changed_answer"},
            name_suffix="CHANGE_SECURITY_ANSWER_NO_TOKEN_POST"
        )

    @task(1)
    def csrf_update_basket_item_put(self):
        item_id = random.randint(1, 50)
        quantity = random.randint(1, 5)
        self._send_csrf_request(
            "put",
            "basket_item",
            path_format_args={"item_id": item_id},
            data={"quantity": quantity},
            name_suffix="UPDATE_BASKET_ITEM_NO_TOKEN_PUT"
        )

    @task(1)
    def csrf_delete_basket_item_delete(self):
        item_id = random.randint(1, 50)
        self._send_csrf_request(
            "delete",
            "basket_item",
            path_format_args={"item_id": item_id},
            name_suffix="DELETE_BASKET_ITEM_NO_TOKEN_DELETE"
        )

    @task(1)
    def csrf_submit_privacy_request_post(self):
        self._send_csrf_request(
            "post",
            "privacy_requests",
            data={"subject": "CSRF Privacy Request", "message": "Please delete my data ASAP!"},
            name_suffix="SUBMIT_PRIVACY_REQ_NO_TOKEN_POST"
        )

    @task(1)
    def csrf_submit_feedback_post(self):
        self._send_csrf_request(
            "post",
            "feedbacks",
            data={"comment": "CSRF Feedback Comment", "rating": random.randint(1, 5)},
            name_suffix="SUBMIT_FEEDBACK_NO_TOKEN_POST"
        )

    @task(1)
    def csrf_submit_review_post(self):
        product_id = random.randint(1, 20)
        self._send_csrf_request(
            "post",
            "product_reviews_by_id",
            path_format_args={"product_id": product_id},
            data={"message": "CSRF Review Message", "author": "CSRF Reviewer"},
            name_suffix="SUBMIT_REVIEW_NO_TOKEN_POST"
        )



    ##
    @task(3)
    def csrf_add_to_cart_post(self):
        item_id = random.randint(1, 50)
        quantity = random.randint(1, 5)
        self._send_request(
            "post",
            "add_to_cart",
            data={"itemId": item_id, "quantity": quantity},
            name_suffix="CSRF_ADD_TO_CART_NO_TOKEN_POST",
            obfuscation_type="none"
        )

    @task(2)
    def csrf_password_reset_attempt_post(self):
        new_password = "new_password_123"
        self._send_request(
            "post",
            "password_reset",
            data={"email": "user@example.com", "newPassword": new_password},
            name_suffix="CSRF_PASSWORD_RESET_NO_TOKEN_POST",
            obfuscation_type="none"
        )

    @task(1)
    def csrf_admin_action_attempt_post(self):
        target_user_id = random.randint(1, 10)
        self._send_request(
            "post",
            "admin_panel",
            path_format_args={"user_id": target_user_id},
            data={"action": "deleteUser", "userId": target_user_id},
            name_suffix="CSRF_ADMIN_DELETE_USER_NO_TOKEN_POST",
            obfuscation_type="none"
        )

    @task(1)
    def csrf_with_invalid_token_post(self):
        item_id = random.randint(1, 50)
        quantity = random.randint(1, 5)
        self._send_request(
            "post",
            "add_to_cart",
            data={"itemId": item_id, "quantity": quantity},
            headers=AppConfig.CSRF_HEADERS,
            name_suffix="CSRF_ADD_TO_CART_INVALID_TOKEN_POST",
            obfuscation_type="none"
        )

    @task(1)
    def csrf_update_product_put(self):
        product_id = random.randint(1, 10)
        self._send_request(
            "put",
            "update_product",
            path_format_args={"product_id": product_id},
            data={"name": "CSRF Updated Product", "price": 99.99},
            name_suffix="CSRF_UPDATE_PRODUCT_NO_TOKEN_PUT",
            obfuscation_type="none"
        )

    @task(1)
    def csrf_delete_product_delete(self):
        product_id = random.randint(1, 10)
        self._send_request(
            "delete",
            "delete_product",
            path_format_args={"product_id": product_id},
            name_suffix="CSRF_DELETE_PRODUCT_NO_TOKEN_DELETE",
            obfuscation_type="none"
        )

