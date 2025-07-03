class AppConfig:
    """
    Centralized configuration for application endpoints, parameters, and attack payloads.
    This makes the code more extensible and easier to manage.
    """
    # Base URL for the application (can be overridden by Locust CLI --host)
    BASE_HOST = "http://localhost:8080"

    # --- Task Weights (moved to config) ---
    USER_TASK_WEIGHTS = { # ints only
        "LegitimateUser": 10,
        "SQLiAttacker":0,
        "XSSAttacker": 0,
        "DirectoryTraversalAttacker": 0,
        "EnumerationAttacker": 0,
        "CSRFAttacker": 0,
    }
    

    # --- User Agents ---
    # A selection of legitimate-looking User-Agent strings
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/125.0.6422.80 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/125.0.0.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
    ]

    # --- Endpoints ---
    # A dictionary mapping logical endpoint names to their actual paths.
    ENDPOINTS = {
        "homepage": "/",
        "product_search": "/rest/products/search",
        "product_details": "/rest/products/{product_id}",
        "user_login": "/rest/user/login",
        "user_register": "/api/Users/",
        "feedback": "/api/Feedbacks/",
        "file_download": "/download",
        "image_loader": "/images",
        "user_profile": "/api/Users/{user_id}",
        "password_reset": "/rest/user/forgot-password",
        "admin_panel": "/admin",
        "add_to_cart": "/rest/cart/add",
        "view_cart": "/rest/cart",
        "update_cart_item": "/rest/cart/item/{product_id}",
        "remove_from_cart": "/rest/cart/item/{product_id}",
        "create_order": "/rest/orders/",
        "update_product": "/rest/products/{product_id}",
        "delete_product": "/rest/products/{product_id}",
    }

    # --- Common Query Parameters ---
    COMMON_QUERY_PARAMS = ["q", "query", "id", "category", "search", "term", "file", "path", "data"]

    # Common usernames for enumeration attempts
    COMMON_USERNAMES = ["admin", "root", "test", "user", "guest", "administrator", "support"]

    # CSRF related configurations
    CSRF_TOKEN_NAMES = ["_csrf", "csrf_token", "authenticity_token"]
    CSRF_HEADERS = {"X-CSRF-TOKEN": "invalid_token"}


