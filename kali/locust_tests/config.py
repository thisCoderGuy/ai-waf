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
        "DirectoryTraversalAttacker": 10,
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
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
 # Android Chrome - Samsung Internet
    "Mozilla/5.0 (Linux; Android 13; SAMSUNG SM-S918U) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/24.0 Chrome/125.0.0.0 Mobile Safari/537.36",
    
    # Google Pixel Android
    "Mozilla/5.0 (Linux; Android 14; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.0 Mobile Safari/537.36",

    # iPhone Safari (older iOS)
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",

    # iPad Chrome
    "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) CriOS/125.0.6422.76 Mobile/15E148 Safari/604.1",

    # Edge on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",

    # Brave Browser on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Brave/125.1.63.162",

    # Headless Chrome (automated tools like Puppeteer)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/125.0.0.0 Safari/537.36",

    # curl (CLI tool)
    "curl/8.6.0",

    # Wget
    "Wget/1.21.3",

    # YandexBot
    "Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)",

    # Baidu Spider
    "Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)"
    ]

    # --- Endpoints ---
    # A dictionary mapping logical endpoint names to their actual paths.
    ENDPOINTS = {
        "homepage": "/",

        # User Management
        "user_register": "/api/Users/",
        "user_profile": "/api/Users/{user_id}",
        "user_profile": "/profile",
        
        # Product Management        
        "manage_products": "/api/Products/",
        "manage_product": "/api/Products/{product_id}",

        # Feedback Management
        "feedbacks": "/api/Feedbacks/",
        "feedback": "/api/Feedbacks/{user_id}",

        # Shopping Cart
        "basket_items": "/api/BasketItems",
        "basket_item": "/api/BasketItems/{item_id}",

        # Challenges
        "challenges": "/api/Challenges",
        "challenge": "/api/Challenges/{challenge_id}",
        "challenge_continue": "/rest/continue-code",

        # Complaint
        "complaints": "/api/Complaints",
        "complaint": "/api/Complaints/{complaint_id}",

        # Recyclings
        "recycles": "/api/Recycles",        
        "recycle": "/api/Recycles/{recycle_id}",

        # Security Questions
        "security_questions": "/api/SecurityQuestions",
        "security_question": "/api/SecurityQuestions/{security_question_id}",

        # Security Answers
        "security_answers": "/api/SecurityAnswers",
        "security_answer": "/api/SecurityAnswers/{security_answer_id}",

        # Address
        "addresses": "/api/Addresss",
        "address": "/api/Addresss/{address_id}",

        # Privacy Requests
        "privacy_requests": "/api/PrivacyRequests",
        "privacy_request": "/api/PrivacyRequests/{privacy_request_id}",

        # Payment Cards
        "cards": "/api/Cards",
        "card": "/api/Cards/{card_id}",

        # ORders
        "orders": "/api/Orders",
        "order": "/api/Orders/{order_id}",

        # Quantities
        "quantities": "/api/Quantitys",
        "quantity": "/api/Quantitys/{quantity_id}",

        # Authentication
        "user_login": "/rest/user/login",
        "registration": "/rest/user/registration",
        "authentication": "/rest/user/authentication",
        "change_password": "/rest/user/change-password",
        "reset_password": "/rest/user/reset-password",
        "password_reset": "/rest/user/forgot-password",
        "who_am_i": "/rest/user/whoami",

        # Product Management
        "product_search": "/rest/products/search",
        "product_details": "/rest/products/{product_id}",

         # Product Reviews
        "product_reviews": "/rest/products/reviews",
        "product_review": "/rest/products/{product_id}/reviews/",


        # Basket checkout
        "basket_checkout": "/rest/basket/{basket_id}/checkout/",
        "basket": "/rest/basket/{basket_id}",

        # Order
        "track_order": "/rest/track-order/{order_id}",
        "track_order_result": "/rest/track-order/result",
        "create_order": "/rest/orders/",

        # Cart
        "add_to_cart": "/rest/cart/add",
        "view_cart": "/rest/cart",
        "update_cart_item": "/rest/cart/item/{product_id}",
        "remove_from_cart": "/rest/cart/item/{product_id}",

        # Misc.
        "save_login_ip": "/rest/saveLoginIp",
        "file_download": "/download",
        "image_loader": "/images",
        "admin_panel": "/admin",
        "captcha": "/rest/captcha/",
        "image_captcha": "/rest/image-captcha/",
        "file_upload": "/file-upload",
        "profile_image_file": "/profile/image/file",
        "profile_image_url": "/profile/image/url",
        "ftp_access": "/ftp/{path}",
        "encryption_keys": "/encryptionkeys/{file}",
        "assets": "/assets/{file}",

         # B2B Operations
        "b2b_orders": "/b2b/v2/orders",
        
        # Application Info
        "app_version": "/rest/admin/application-version",
        "app_config": "/rest/admin/application-configuration",
        "languages": "/rest/languages",
        
         # Memories/Photos
        "memories": "/rest/memories",
        "memory_detail": "/rest/memories/{id}",
        
        # Wallet & Deluxe
        "wallet": "/rest/wallet/balance",
        "deluxe_membership": "/rest/deluxe-membership",
        
        # Login IP tracking
        "save_login_ip": "/rest/saveLoginIp",
        "last_login_ip": "/rest/user/last-login-ip",
        
        # Two-Factor Authentication
        "two_factor_auth": "/rest/user/two-factor-auth",
        "two_factor_verify": "/rest/user/two-factor-auth/verify",
        
        # Data Export
        "data_export": "/rest/user/data-export",
        
        # Chatbot
        "chatbot": "/rest/chatbot",
        "chatbot_status": "/rest/chatbot/status",
        
        # Error/Debug endpoints
        "redirect": "/redirect",
        "error_500": "/rest/error/500",
        
        # Metrics (if enabled)
        "metrics": "/metrics",
        "profile": "/profile",
        
        # WebSocket
        "websocket": "/socket.io/",
        
        
    }

    # --- Common Query Parameters ---
    COMMON_QUERY_PARAMS = ["q", "query", "id", "category", "search", "term", "file", "path", "data"]

    # Common usernames for enumeration attempts
    COMMON_USERNAMES = ["admin", "root", "test", "user", "guest", "administrator", "support"]

    # CSRF related configurations
    CSRF_TOKEN_NAMES = ["_csrf", "csrf_token", "authenticity_token"]
    CSRF_HEADERS = {"X-CSRF-TOKEN": "invalid_token"}


