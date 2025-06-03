import random
from locust import task

from ..base_attacker import BaseAttacker
from ..config import AppConfig # Needed for common usernames, endpoints

#TODO Add obfuscation


class EnumerationAttacker(BaseAttacker):
    """
    Simulates attempts to enumerate valid users, product IDs, or other resources.
    """
    @task(3)
    def user_enumeration_login_post(self):
        username = random.choice(AppConfig.COMMON_USERNAMES)
        self._send_request(
            "post",
            "user_login",
            data={"email": f"{username}@example.com", "password": "wrongpassword"},
            name_suffix="USER_ENUM_LOGIN_POST",
            obfuscation_type="none"
        )

    @task(3)
    def user_enumeration_password_reset_post(self):
        username = random.choice(AppConfig.COMMON_USERNAMES)
        self._send_request(
            "post",
            "password_reset",
            data={"email": f"{username}@example.com"},
            name_suffix="USER_ENUM_PASSWORD_RESET_POST",
            obfuscation_type="none"
        )

    @task(2)
    def product_id_enumeration_get(self):
        product_id = random.randint(1, 100)
        self._send_request(
            "get",
            "product_details",
            path_format_args={"product_id": product_id},
            name_suffix="PRODUCT_ENUM_GET",
            obfuscation_type="none"
        )

    @task(1)
    def admin_page_enumeration_get(self):
        admin_paths = ["/admin", "/dashboard", "/controlpanel", "/wp-admin", "/phpmyadmin"]
        path = random.choice(admin_paths)
        self.client.get(path, name=f"{path}/[GET_ADMIN_PAGE_ENUM]")

    @task(1)
    def user_id_enumeration_delete(self):
        user_id = random.randint(1, 20)
        self._send_request(
            "delete",
            "user_profile",
            path_format_args={"user_id": user_id},
            name_suffix="USER_ENUM_DELETE",
            obfuscation_type="none"
        )