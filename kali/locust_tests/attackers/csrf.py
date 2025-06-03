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

