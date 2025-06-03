import random
from locust import TaskSet, task # Import TaskSet

from .config import AppConfig

class LegitimateUser(TaskSet):
    """
    Simulates a typical, non-malicious user browsing the application,
    now with enhanced shopping cart interactions.
    """
    def on_start(self):
        self.cart_items = {} # {product_id: quantity}

    @task(30) # Was 3, now 3 * 10
    def view_homepage(self):
        self.client.get(AppConfig.ENDPOINTS["homepage"])

    @task(20) # Was 2, now 2 * 10
    def browse_products(self):
        search_terms = ["apple", "juice", "lemon", "banana", "orange", "milk", "bread", "coffee"]
        query = random.choice(search_terms)
        self.client.get(f"{AppConfig.ENDPOINTS['product_search']}?q={query}")

    @task(10) # Was 1, now 1 * 10
    def view_product_details(self):
        product_id = random.randint(1, 20)
        self.client.get(AppConfig.ENDPOINTS["product_details"].format(product_id=product_id))

    @task(10) # Was 1, now 1 * 10
    def login_attempt(self):
        self.client.post(AppConfig.ENDPOINTS["user_login"], json={
            "email": "test@example.com",
            "password": "password123"
        })

    @task(10) # Was 1, now 1 * 10
    def register_user(self):
        email = f"user{random.randint(1000, 9999)}@example.com"
        password = "securepassword" + str(random.randint(1, 1000))
        self.client.post(AppConfig.ENDPOINTS["user_register"], json={
            "email": email,
            "password": password,
            "passwordRepeat": password,
            "securityAnswer": "My pet's name"
        })

    @task(20) # Was 2, now 2 * 10
    def add_item_to_cart(self):
        product_id = random.randint(1, 20)
        quantity = random.randint(1, 3)
        self.client.post(AppConfig.ENDPOINTS["add_to_cart"], json={
            "productId": product_id,
            "quantity": quantity
        })
        self.cart_items[product_id] = self.cart_items.get(product_id, 0) + quantity

    @task(15) # Was 1.5, now 1.5 * 10
    def view_shopping_cart(self):
        self.client.get(AppConfig.ENDPOINTS["view_cart"])

    @task(10) # Was 1, now 1 * 10
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

    @task(8) # Was 0.8, now 0.8 * 10
    def remove_item_from_cart(self):
        if not self.cart_items:
            return

        product_id = random.choice(list(self.cart_items.keys()))
        self.client.delete(
            AppConfig.ENDPOINTS["remove_from_cart"].format(product_id=product_id)
        )
        del self.cart_items[product_id]

    @task(5) # Was 0.5, now 0.5 * 10
    def create_order(self):
        if not self.cart_items:
            return

        items_payload = [{"productId": pid, "quantity": qty} for pid, qty in self.cart_items.items()]
        if not items_payload:
            return

        self.client.post(AppConfig.ENDPOINTS["create_order"], json={"items": items_payload, "customerName": "Legit User"})
        self.cart_items = {}


    @task(5) # Was 0.5, now 0.5 * 10
    def update_profile(self):
        user_id = random.randint(1, 5)
        self.client.put(
            AppConfig.ENDPOINTS["user_profile"].format(user_id=user_id),
            json={"email": f"updated_user{user_id}@example.com", "address": "123 Main St"}
        )