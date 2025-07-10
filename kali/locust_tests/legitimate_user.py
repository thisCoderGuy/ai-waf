import random
from locust import TaskSet, task # Import TaskSet

from .config import AppConfig

class LegitimateUser(TaskSet):
    """
    Simulates a typical, non-malicious user browsing the application,
    now with enhanced shopping cart interactions.
    """

    def on_start(self):
        """
        Called when a LegitimateUser starts. Always sets a random legitimate User-Agent.
        """
        self.client.headers["User-Agent"] = random.choice(AppConfig.USER_AGENTS)
        self.cart_items = {} # {product_id: quantity}

    @task(30) 
    def view_homepage(self):
        self.client.get(AppConfig.ENDPOINTS["homepage"])

    @task(20) 
    def browse_products(self):
        search_terms = ["apple", "juice", "lemon", "banana", "orange", "milk", "bread", "coffee"]
        query = random.choice(search_terms)
        self.client.get(f"{AppConfig.ENDPOINTS['product_search']}?q={query}")

    @task(10) 
    def view_product_details(self):
        product_id = random.randint(1, 20)
        self.client.get(AppConfig.ENDPOINTS["product_details"].format(product_id=product_id))

    @task(10)
    def login_attempt(self):
        self.client.post(AppConfig.ENDPOINTS["user_login"], json={
            "email": "test@example.com",
            "password": "password123"
        })

    @task(10)
    def register_user(self):
        email = f"user{random.randint(1000, 9999)}@example.com"
        password = "securepassword" + str(random.randint(1, 1000))
        self.client.post(AppConfig.ENDPOINTS["user_register"], json={
            "email": email,
            "password": password,
            "passwordRepeat": password,
            "securityAnswer": "My pet's name"
        })

    @task(20) 
    def add_item_to_cart(self):
        product_id = random.randint(1, 20)
        quantity = random.randint(1, 3)
        self.client.post(AppConfig.ENDPOINTS["add_to_cart"], json={
            "productId": product_id,
            "quantity": quantity
        })
        self.cart_items[product_id] = self.cart_items.get(product_id, 0) + quantity

    @task(15) 
    def view_shopping_cart(self):
        self.client.get(AppConfig.ENDPOINTS["view_cart"])

    @task(10) 
    def update_item_in_cart(self):
        if not self.cart_items:
            return

        product_id = random.choice(list(self.cart_items.keys()))
        new_quantity = random.randint(1, 5)
        self.client.put(
            AppConfig.ENDPOINTS["update_cart_item"].format(product_id=product_id),
            json={"quantity": new_quantity}
        )
        self.cart_items[product_id] = new_quantity

    @task(8) 
    def remove_item_from_cart(self):
        if not self.cart_items:
            return

        product_id = random.choice(list(self.cart_items.keys()))
        self.client.delete(
            AppConfig.ENDPOINTS["remove_from_cart"].format(product_id=product_id)
        )
        del self.cart_items[product_id]

    @task(5)
    def create_order(self):
        if not self.cart_items:
            return

        items_payload = [{"productId": pid, "quantity": qty} for pid, qty in self.cart_items.items()]
        if not items_payload:
            return

        self.client.post(AppConfig.ENDPOINTS["create_order"], json={"items": items_payload, "customerName": "Legit User"})
        self.cart_items = {}


    @task(5) 
    def update_profile(self):
        user_id = random.randint(1, 5)
        self.client.put(
            AppConfig.ENDPOINTS["user_profile"].format(user_id=user_id),
            json={"email": f"updated_user{user_id}@example.com", "address": "123 Main St"}
        )

    @task(5)
    def submit_feedback(self):
        self.client.post(AppConfig.ENDPOINTS["feedbacks"], json={
            "comment": f"Great site, found a bug! {random.randint(1,1000)}",
            "rating": random.randint(1, 5)
        })

    @task(3)
    def view_all_challenges(self):
        self.client.get(AppConfig.ENDPOINTS["challenges"])

    @task(2)
    def get_specific_challenge_details(self):
        challenge_id = random.randint(1, 50) # Assuming 50 challenges for example
        self.client.get(AppConfig.ENDPOINTS["challenge"].format(challenge_id=challenge_id))

    @task(5)
    def upload_complaint_file(self):
        # This is a POST request, often with multipart/form-data.
        # For Locust, this can be simulated with files parameter.
        # Actual content could be malicious.
        file_content = b"This is a test complaint from Locust."
        self.client.post(
            AppConfig.ENDPOINTS["file_upload"],
            files={"file": ("complaint.txt", file_content, "text/plain")}
        )

    @task(2)
    def get_security_questions(self):
        self.client.get(AppConfig.ENDPOINTS["user_security_questions_all"])

    @task(1)
    def get_specific_security_question(self):
        question_id = random.randint(1, 5) # Assuming a few questions exist
        self.client.get(AppConfig.ENDPOINTS["user_security_question_by_id"].format(security_question_id=question_id))

    @task(5)
    def manage_payment_cards(self):
        # Example: Add a new card (POST)
        self.client.post(AppConfig.ENDPOINTS["cards"], json={
            "fullName": "Locust User",
            "cardNum": f"123456789012{random.randint(1000,9999)}",
            "expMonth": random.randint(1,12),
            "expYear": random.randint(2025,2030)
        })
        # Example: Get all cards (GET)
        # self.client.get(AppConfig.ENDPOINTS["cards"])

    @task(5)
    def manage_addresses(self):
        # Example: Add a new address (POST)
        self.client.post(AppConfig.ENDPOINTS["addresses"], json={
            "fullName": "Locust User",
            "country": "USA",
            "zipCode": f"{random.randint(10000, 99999)}",
            "streetAddress": f"{random.randint(1,100)} Test St",
            "city": "Locustville",
            "state": "FL"
        })
        # Example: Get all addresses (GET)
        # self.client.get(AppConfig.ENDPOINTS["addresses"])

    @task(3)
    def track_an_order(self):
        # In Juice Shop, order IDs are specific and not sequential for tracking without auth.
        # This would usually require a valid order ID from a prior purchase.
        # For simulation, use a placeholder or known order ID for the challenge.
        order_id = "some_valid_order_id_from_challenge" # Replace with actual dynamic or known IDs
        self.client.get(AppConfig.ENDPOINTS["track_order"].format(order_id=order_id))

    @task(1)
    def get_application_version(self):
        self.client.get(AppConfig.ENDPOINTS["app_version"])

    @task(1)
    def get_metrics(self):
        self.client.get(AppConfig.ENDPOINTS["metrics"])

    @task(2)
    def trigger_error(self):
        # This is a challenge to trigger a 500 error
        self.client.get(AppConfig.ENDPOINTS["error_500_trigger"])

    @task(2)
    def use_chatbot(self):
        # Simulate interaction with the chatbot
        self.client.post(AppConfig.ENDPOINTS["customer_chatbot"], json={"query": "Hello, what are your opening hours?"})

    @task(1)
    def check_robots_txt(self):
        self.client.get(AppConfig.ENDPOINTS["robots_txt"])

    @task(1)
    def check_sitemap_xml(self):
        self.client.get(AppConfig.ENDPOINTS["sitemap_xml"])

    @task(3)
    def view_product_reviews(self):
        product_id = random.randint(1, 20)
        self.client.get(AppConfig.ENDPOINTS["product_reviews_by_id"].format(product_id=product_id))

    @task(1)
    def get_languages(self):
        self.client.get(AppConfig.ENDPOINTS["languages"])

    @task(1)
    def get_wallet_balance(self):
        # This typically requires authentication
        self.client.get(AppConfig.ENDPOINTS["wallet_balance"])

    @task(1)
    def visit_ftp_directory(self):
        # Path traversal challenge, usually involves specific file names
        self.client.get(AppConfig.ENDPOINTS["ftp_access"].format(path="/")) # Accessing root of FTP, or try "files" or "../"

    @task(1)
    def get_security_txt(self):
        self.client.get(AppConfig.ENDPOINTS["security_txt"])