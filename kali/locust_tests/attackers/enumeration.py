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
        """
        Attempts to enumerate users by observing login response differences
        for existing vs. non-existing emails.
        """
        username_or_email = random.choice(AppConfig.COMMON_USERNAMES)
        # Use a common password to avoid password strength issues, focus on username enum
        self._send_request(
            "post",
            "user_login",
            data={"email": f"{username_or_email}@example.com", "password": "wrongpassword"},
            name_suffix="USER_ENUM_LOGIN_POST",
            obfuscation_type="none" # Enumeration typically doesn't obfuscate payloads
        )
        # Also test with a non-existent username
        self._send_request(
            "post",
            "user_login",
            data={"email": f"nonexistent_user_{random.randint(10000,99999)}@example.com", "password": "wrongpassword"},
            name_suffix="USER_ENUM_LOGIN_NONEXISTENT_POST",
            obfuscation_type="none"
        )


    @task(3)
    def user_enumeration_password_reset_post(self):
        """
        Attempts to enumerate users by observing password reset response differences.
        """
        username_or_email = random.choice(AppConfig.COMMON_USERNAMES)
        self._send_request(
            "post",
            "password_reset",
            data={"email": f"{username_or_email}@example.com"},
            name_suffix="USER_ENUM_PASSWORD_RESET_POST",
            obfuscation_type="none"
        )
        # Also test with a non-existent username
        self._send_request(
            "post",
            "password_reset",
            data={"email": f"nonexistent_user_{random.randint(10000,99999)}@example.com"},
            name_suffix="USER_ENUM_PASSWORD_RESET_NONEXISTENT_POST",
            obfuscation_type="none"
        )


    @task(2)
    def product_id_enumeration_get(self):
        """
        Attempts to enumerate product IDs by iterating through common ranges.
        """
        product_id = random.randint(1, 100) # Common range for existing products
        self._send_request(
            "get",
            "product_details",
            path_format_args={"product_id": product_id},
            name_suffix="PRODUCT_ENUM_GET",
            obfuscation_type="none"
        )
        # Attempt to access a likely non-existent product ID
        self._send_request(
            "get",
            "product_details",
            path_format_args={"product_id": random.randint(10000, 20000)}, # Large ID, likely non-existent
            name_suffix="PRODUCT_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )


    @task(1)
    def admin_page_enumeration_get(self):
        """
        Attempts to discover hidden or sensitive admin/management pages.
        """
        admin_paths = [
            "/admin", "/administration", "/dashboard", "/controlpanel",
            "/users", "/products/manage", "/settings", "/config", "/metrics",
            "/swagger.json", "/api-docs/", "/openapi.json",
            "/robots.txt", "/sitemap.xml", # Often contain sensitive paths
            "/backup", "/logs", "/debug", "/test",
            "/phpmyadmin", "/wp-admin", # Common default paths for other systems
            "/rest/admin/application-version",
            "/rest/admin/application-configuration",
            "/rest/admin/debug", # Specific Juice Shop debug endpoint
            "/rest/admin/users", # Hypothetical admin API endpoint for users
        ]
        path_to_try = random.choice(admin_paths)
        # For direct path access, use client.get directly as _send_request expects AppConfig.ENDPOINTS keys
        self.client.get(path_to_try, name=f"{path_to_try}/[GET_ADMIN_PAGE_ENUM]")


    @task(1)
    def user_id_enumeration_delete(self):
        """
        Attempts to enumerate user IDs by trying to delete them.
        Observes response differences (e.g., 401 vs 404 vs 200).
        """
        user_id = random.randint(1, 20) # Common range for existing users (especially low IDs like admin)
        self._send_request(
            "delete",
            "user_profile",
            path_format_args={"user_id": user_id},
            name_suffix="USER_ENUM_DELETE",
            obfuscation_type="none"
        )
        # Attempt to delete a likely non-existent user ID
        self._send_request(
            "delete",
            "user_profile",
            path_format_args={"user_id": random.randint(1000, 2000)}, # Large ID, likely non-existent
            name_suffix="USER_ENUM_DELETE_NONEXISTENT",
            obfuscation_type="none"
        )

    # --- NEW ENUMERATION TASKS ---

    @task(2)
    def order_id_enumeration_get(self):
        """
        Attempts to enumerate order IDs by trying to retrieve them.
        """
        order_id = random.randint(1, 50) # Common range for existing orders
        self._send_request(
            "get",
            "order",
            path_format_args={"order_id": order_id},
            name_suffix="ORDER_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "order",
            path_format_args={"order_id": random.randint(10000, 20000)}, # Non-existent
            name_suffix="ORDER_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )

    @task(2)
    def feedback_id_enumeration_get(self):
        """
        Attempts to enumerate feedback IDs.
        """
        feedback_id = random.randint(1, 100)
        self._send_request(
            "get",
            "feedback_by_id",
            path_format_args={"feedback_id": feedback_id},
            name_suffix="FEEDBACK_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "feedback_by_id",
            path_format_args={"feedback_id": random.randint(10000, 20000)},
            name_suffix="FEEDBACK_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )

    @task(1)
    def challenge_id_enumeration_get(self):
        """
        Attempts to enumerate challenge IDs.
        """
        challenge_id = random.randint(1, 110) # Juice Shop has ~110 challenges
        self._send_request(
            "get",
            "challenge",
            path_format_args={"challenge_id": challenge_id},
            name_suffix="CHALLENGE_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "challenge",
            path_format_args={"challenge_id": random.randint(500, 1000)},
            name_suffix="CHALLENGE_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )

    @task(1)
    def file_enumeration_get(self):
        """
        Attempts to enumerate common sensitive files.
        """
        common_files = [
            "package.json", "package-lock.json", ".env", ".git/config",
            "README.md", "LICENSE", "robots.txt", "sitemap.xml",
            "data/access.log", # Hypothetical log file
            "data/users.json", # Hypothetical user data file
            "data/backup.zip", # Hypothetical backup file
            "admin/config.php", # Common for PHP apps
            "web.config", # Common for .NET apps
        ]
        file_to_try = random.choice(common_files)
        # Use client.get directly for arbitrary paths not in AppConfig.ENDPOINTS
        self.client.get(f"/{file_to_try}", name=f"/{file_to_try}/[GET_FILE_ENUM]")

        # Also try via file_download endpoint if it takes a filename
        self._send_request(
            "get",
            "file_download",
            params={"file": file_to_try},
            name_suffix="FILE_DOWNLOAD_ENUM_QUERY",
            obfuscation_type="none"
        )

    @task(1)
    def security_question_enumeration_get(self):
        """
        Attempts to enumerate security question IDs.
        """
        question_id = random.randint(1, 10) # Assuming a small number of questions
        self._send_request(
            "get",
            "security_question_by_id",
            path_format_args={"security_question_id": question_id},
            name_suffix="SEC_QUESTION_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "security_question_by_id",
            path_format_args={"security_question_id": random.randint(100, 200)},
            name_suffix="SEC_QUESTION_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )

    @task(1)
    def address_id_enumeration_get(self):
        """
        Attempts to enumerate address IDs.
        """
        address_id = random.randint(1, 50)
        self._send_request(
            "get",
            "address",
            path_format_args={"address_id": address_id},
            name_suffix="ADDRESS_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "address",
            path_format_args={"address_id": random.randint(10000, 20000)},
            name_suffix="ADDRESS_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )

    @task(1)
    def card_id_enumeration_get(self):
        """
        Attempts to enumerate payment card IDs.
        """
        card_id = random.randint(1, 50)
        self._send_request(
            "get",
            "card",
            path_format_args={"card_id": card_id},
            name_suffix="CARD_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "card",
            path_format_args={"card_id": random.randint(10000, 20000)},
            name_suffix="CARD_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )

    @task(1)
    def memory_id_enumeration_get(self):
        """
        Attempts to enumerate memory (photo) IDs.
        """
        memory_id = random.randint(1, 50)
        self._send_request(
            "get",
            "memory_detail",
            path_format_args={"id": memory_id},
            name_suffix="MEMORY_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "memory_detail",
            path_format_args={"id": random.randint(10000, 20000)},
            name_suffix="MEMORY_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )

    @task(1)
    def privacy_request_id_enumeration_get(self):
        """
        Attempts to enumerate privacy request IDs.
        """
        privacy_id = random.randint(1, 20)
        self._send_request(
            "get",
            "privacy_request",
            path_format_args={"privacy_request_id": privacy_id},
            name_suffix="PRIVACY_REQ_ENUM_GET",
            obfuscation_type="none"
        )
        self._send_request(
            "get",
            "privacy_request",
            path_format_args={"privacy_request_id": random.randint(1000, 2000)},
            name_suffix="PRIVACY_REQ_ENUM_NONEXISTENT_GET",
            obfuscation_type="none"
        )