class AppConfig:
    """
    Centralized configuration for application endpoints, parameters, and attack payloads.
    This makes the code more extensible and easier to manage.
    """
    # Base URL for the application (can be overridden by Locust CLI --host)
    BASE_HOST = "http://localhost:8080"

    # --- Task Weights (moved to config) ---
    USER_TASK_WEIGHTS = { # ints only
        "LegitimateUser": 0,
        "SQLiAttacker": 0,
        "XSSAttacker": 0,
        "DirectoryTraversalAttacker": 0,
        "EnumerationAttacker": 10,
        "CSRFAttacker": 0,
    }
    

    # --- User Agents ---
    # A selection of legitimate-looking User-Agent strings
    USER_AGENTS = [
        # Standard browser user agents
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/125.0.6422.80 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/125.0.0.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",        
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

        # Search engine bots 
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
        # YandexBot
        "Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)",
        # Baidu Spider
        "Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)"
    ]

    MALICIOUS_USER_AGENTS = [
        # Headless Chrome (automated tools like Puppeteer)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/125.0.0.0 Safari/537.36",
         # Explicitly indicate automated tools often used for attacks/scraping
        "curl/7.88.1",         # A recent curl version
        "curl/8.6.0",          # Another plausible recent curl version
        "curl/",               # Catches any curl version (good for broad matching)
        "Wget/1.21.3",         # A common Wget version
        "Wget/1.20.3",         # Another plausible Wget version
        "Wget/",               # Catches any Wget version
        "HeadlessChrome/",     # Catches any Headless Chrome version
        "sqlmap/1.7.9",        # Specific sqlmap version
        "sqlmap/",             # Catches any sqlmap version
        "Nmap Scripting Engine", # Nmap, often without a direct version in the UA string, but identifiable by this phrase
        "Acunetix/14.8.230718105", # Example Acunetix version
        "Acunetix/",           # Catches any Acunetix version
        "Nessus/",             # Nessus, often generic or with less specific versions in the UA
        "Nikto/2.1.6",         # A common Nikto version
        "Nikto/",              # Catches any Nikto version
        "BurpSuite Professional/2024.5.4", # Example Burp Suite version
        "BurpSuite Free/2024.5.4", # Example Burp Suite Free version
        "BurpSuite",           # Catches any Burp Suite version
        "OpenVAS/",            # OpenVAS, often generic in the UA
        "Python-requests/2.31.0", # Common Python requests version
        "Go-http-client/1.1",  # Common Go HTTP client version
        "Java/1.8.0_301",      # Specific Java version
        "Java/1.13.0_201",              
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322)", # Very old/suspicious IE
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)", # Another old/suspicious IE
        "sqlmap/1.7.9#dev (http://sqlmap.org)",
        "AhrefsBot/7.1",
        "SemrushBot/7.0",
        "Mozilla/5.0 (compatible; BLEXBot/1.1; +http://webmeup.com/)",
        "Mozilla/5.0 (compatible; MJ12bot/v1.4.8; http://mj12bot.com/)",
        "Mozilla/5.0 (compatible; YisouSpider/4.0)",
        "Sogou web spider/4.0(+http://www.sogou.com/docs/help/webmasters.htm#07)",
        "Exabot/3.0",
        "GuzzleHttp/7.2",
        "Python-urllib/3.10",      

    ]

    # --- Endpoints ---
    # A dictionary mapping logical endpoint names to their actual paths.
    ENDPOINTS = {
    # Core Application & Static File Serving (Server-side rendered pages / root)
    "homepage": "/",                       # GET: Serves the main index.html
    "admin_panel_html": "/admin",          # GET: Serves the admin panel HTML page
    "profile_html": "/profile",            # GET: Serves the user profile HTML page

    # User Management & Authentication (Backend API)
    "user_register": "/api/Users/",             # POST: Register a new user
    "user_profile": "/api/Users/{user_id}",     # GET: Retrieve user by ID; PATCH/PUT: Update user; DELETE: Delete user
    "user_login": "/rest/user/login",           # POST: Authenticate user, returns JWT
    "registration": "/rest/user/registration",  # POST: Alternative registration path, often redirects or maps to /api/Users/
    "authentication": "/rest/user/authentication", # POST: General authentication endpoint, might be redundant with login or for specific challenge
    "change_password": "/rest/user/change-password", # POST: Change password for currently authenticated user
    "reset_password": "/rest/user/reset-password", # POST: Reset password, typically used with security answers
    "password_reset_forgot": "/rest/user/forgot-password", # POST: Initiate forgot password flow, might involve email/security Q
    "who_am_i": "/rest/user/whoami",            # GET: Get details of the currently authenticated user
    "save_login_ip": "/rest/saveLoginIp",       # POST/GET: Endpoint related to tracking or saving user login IP addresses
    "last_login_ip": "/rest/user/last-login-ip", # GET: Retrieve the last logged-in IP address for the current user
    "two_factor_auth_setup": "/rest/user/two-factor-auth", # POST: Setup 2FA for a user; GET: Check 2FA status
    "two_factor_auth_verify": "/rest/user/two-factor-auth/verify", # POST: Verify a 2FA token
    "data_export": "/rest/user/data-export",    # GET: Export user data (related to GDPR or data exfiltration challenges)
    "user_data_import": "/rest/user/data-import", # POST: Import user data (less common, but plausible for data handling challenges)
    "user_security_questions_all": "/api/SecurityQuestions", # GET: Retrieve all available security questions
    "user_security_question_by_id": "/api/SecurityQuestions/{security_question_id}", # GET: Retrieve a specific security question; PATCH/PUT: Update (admin); DELETE: Delete (admin)
    "user_security_answers_all": "/api/SecurityAnswers", # GET: Retrieve all security answers (highly privileged, admin); POST: Submit a new security answer (e.g., during registration)
    "user_security_answer_by_id": "/api/SecurityAnswers/{security_answer_id}", # GET: Retrieve a specific security answer; PATCH/PUT: Update; DELETE: Delete (often IDOR vulnerability target)
"security_question_by_id": "/api/SecurityQuestions/{security_question_id}", # For security question ID enumeration
        
    # Product & Review Management (Backend API)
    "manage_products": "/api/Products/",                # GET: Retrieve all products; POST: Create a new product (admin)
    "manage_product": "/api/Products/{product_id}",     # GET: Retrieve product by ID; PATCH/PUT: Update product; DELETE: Delete product
    "product_search": "/rest/products/search",          # GET: Search products using query parameters (e.g., ?q=); common SQLi/XSS target
    "product_details": "/rest/products/{product_id}",   # GET: Retrieve details of a specific product (alternative/legacy path)
    "product_reviews_all": "/rest/products/reviews",    # GET: Retrieve all product reviews; POST: Submit a new product review
    "product_reviews_by_id": "/rest/products/{product_id}/reviews/", # GET: Retrieve reviews for a specific product; POST: Submit review for that product

    # Feedback Management (Backend API)
    "feedbacks": "/api/Feedbacks/",                     # GET: Retrieve all feedback (admin); POST: Submit new feedback
    "feedback_by_id": "/api/Feedbacks/{feedback_id}",   # GET: Retrieve specific feedback; PATCH/PUT: Update; DELETE: Delete (admin)

    # Shopping Cart & Order Management (Backend API)
    "basket_items": "/api/BasketItems",                 # GET: Retrieve items in the current user's basket; POST: Add item to basket
    "basket_item": "/api/BasketItems/{item_id}",        # GET: Retrieve specific basket item; PATCH/PUT: Update quantity; DELETE: Remove item
    "basket_checkout": "/rest/basket/{basket_id}/checkout/", # POST: Finalize the checkout process for a given basket
    "basket": "/rest/basket/{basket_id}",               # GET: Retrieve items in a specific basket
    "orders": "/api/Orders",                            # GET: Retrieve all orders (admin/user specific); POST: Create a new order
    "order": "/api/Orders/{order_id}",                  # GET: Retrieve a specific order; PATCH/PUT: Update (admin); DELETE: Delete (admin)
    "track_order": "/rest/track-order/{order_id}",      # GET: Track an order by ID (often used in Order Tracker challenges)
    "track_order_result": "/rest/track-order/result",   # POST/GET: Endpoint related to displaying or processing order tracking results
    "create_order_rest": "/rest/orders/",               # POST: Alternative/legacy path for creating orders
    "add_to_cart": "/rest/cart/add",                    # POST: Add item to cart (might be alternative/legacy to /api/BasketItems)
    "view_cart": "/rest/cart",                          # GET: View contents of the cart (might be alternative/legacy)
    "update_cart_item": "/rest/cart/item/{product_id}", # PUT/PATCH: Update item quantity in cart (might be alternative/legacy)
    "remove_from_cart": "/rest/cart/item/{product_id}", # DELETE: Remove item from cart (might be alternative/legacy)
    "payment_options": "/api/PaymentOptions/",          # GET: Retrieve available payment options; POST: Add a new payment option
    "wallet_balance": "/rest/wallet/balance",           # GET: Retrieve current user's wallet balance
    "cards": "/api/Cards",                              # GET: Retrieve all saved payment cards; POST: Add a new card
    "card": "/api/Cards/{card_id}",                     # GET: Retrieve specific card; PUT/PATCH: Update; DELETE: Delete (often IDOR target)
    "delivery_methods": "/api/DeliveryMethods/",        # GET: Retrieve available delivery methods
    "addresses": "/api/Addresses",                      # GET: Retrieve all saved addresses; POST: Add a new address
    "address": "/api/Addresses/{address_id}",           # GET: Retrieve specific address; PUT/PATCH: Update; DELETE: Delete (often IDOR target)

    # Complaint & Recycle Management (Backend API)
    "complaints": "/api/Complaints",                    # GET: Retrieve all complaints (admin); POST: Submit a new complaint
    "complaint": "/api/Complaints/{complaint_id}",      # GET: Retrieve specific complaint; PATCH/PUT: Update; DELETE: Delete (admin)
    "recycles": "/api/Recycles",                        # GET: Retrieve all recycle entries; POST: Submit a new recycle entry
    "recycle": "/api/Recycles/{recycle_id}",            # GET: Retrieve specific recycle entry; PATCH/PUT: Update; DELETE: Delete

    # Challenge-Related & Specific Functionality (Backend API)
    "challenges": "/api/Challenges",                    # GET: List all challenges and their solved status
    "challenge": "/api/Challenges/{challenge_id}",      # GET: Retrieve details for a specific challenge
    "challenge_continue": "/rest/continue-code",        # POST: Used for submitting solutions or continuing coding challenges
    "captcha_generation": "/rest/captcha/",             # GET: Generate a new CAPTCHA (for feedback, registration)
    "image_captcha_generation": "/rest/image-captcha/", # GET: Generate an image CAPTCHA
    "file_upload": "/file-upload",                      # POST: General file upload endpoint (common LFI/RCE target)
    "profile_image_file_upload": "/profile/image/file", # POST: Upload a user profile image file
    "profile_image_url_upload": "/profile/image/url",   # POST: Upload profile image via URL (common SSRF target)
    "ftp_access": "/ftp/{path}",                        # GET: Access to the FTP directory (directory listing, path traversal, LFI)
    "encryption_keys_access": "/encryptionkeys/{file}", # GET: Access to encryption key files (sensitive data exposure)
    "assets_access": "/assets/{file}",                  # GET: Access to general static assets (JS, CSS, images)
    "b2b_orders": "/b2b/v2/orders",                     # GET, POST: B2B order operations (often a deprecated interface or specific challenge)
    "app_version": "/rest/admin/application-version",   # GET: Retrieve the application version (information disclosure)
    "app_config": "/rest/admin/application-configuration", # GET: Retrieve application configuration (sensitive info exposure)
    "languages": "/rest/languages",                     # GET: List available languages (information disclosure, potential XXE/XSS via language files)
    "memories": "/rest/memories",                       # GET: Retrieve all memories/photos; POST: Create a new memory
    "memory_detail": "/rest/memories/{id}",             # GET: Retrieve specific memory; PATCH/PUT: Update; DELETE: Delete (often IDOR/XSS target)
    "deluxe_membership": "/rest/deluxe-membership",     # POST: Activate deluxe membership; GET: Check status
    "redirect": "/redirect",                            # GET: Open Redirect vulnerability target
    "error_500_trigger": "/rest/error/500",             # GET: Endpoint to trigger a 500 Internal Server Error (for error handling challenges)
    "metrics": "/metrics",                              # GET: Prometheus metrics endpoint (information disclosure)
    "websocket_connection": "/socket.io/",              # WebSocket endpoint for real-time communication (e.g., chat)
    "customer_chatbot": "/rest/chatbot",                # POST: Interact with the customer support chatbot
    "customer_feedback_backend_target": "/rest/user/feedback", # POST: Backend target for submitting feedback (often a Server-Side Template Injection challenge)
    "admin_route_debug_endpoint": "/rest/admin/debug",  # GET: Hypothetical debug endpoint for admin routes (common challenge)
    "product_file_upload": "/api/Products/{product_id}/upload", # POST: Upload a file related to a product (potential RCE if file type not restricted)
    "coupon_redemption": "/rest/coupon/apply/{coupon_code}", # POST: Apply a coupon code (often for logic flaws or enumeration)
    "file_system_access_challenge": "/rest/admin/file-system", # GET: Endpoint for file system access challenges (path traversal, LFI)
    "data_export_endpoint": "/rest/data/export",        # GET: Export various data (potential for LFI, SSRF, or sensitive data exposure)
    "blockchain_token_sale": "/rest/wallet/token-sale", # POST/GET: Endpoint related to blockchain token sale challenges
    "deprecated_interface": "/rest/deprecated/endpoint", # GET/POST: Placeholder for a deprecated interface challenge
    "robots_txt": "/robots.txt",                         # GET: Standard robots exclusion protocol file
    "sitemap_xml": "/sitemap.xml",                       # GET: Standard sitemap file
    "privacy_requests": "/api/PrivacyRequests",         # GET: Retrieve all privacy requests; POST: Submit a new request
    "privacy_request": "/api/PrivacyRequests/{privacy_request_id}", # GET: Retrieve specific request; PATCH/PUT: Update; DELETE: Delete
    
    # Challenge & Debugging Endpoints (Backend API / Files)
    
    "challenge_snippet_by_key": "/snippets/{challenge_key}", # GET: Retrieve specific challenge snippet
    "recycle_controller": "/api/RecycleController/recycle", # POST: Endpoint for recycle challenges
    
    "swagger_ui_docs": "/api-docs/",            # GET: Swagger UI documentation page
    "swagger_json": "/swagger.json",            # GET: Raw Swagger JSON definition
    "openapi_json": "/openapi.json",            # GET: Raw OpenAPI JSON definition
    "source_map_files": "*.js.map",             # Pattern: JavaScript source map files
    "package_json": "/package.json",            # GET: Project's package.json
    "readme_md": "/README.md",                  # GET: Project's README.md
    "license_file": "/LICENSE",                 # GET: Project's LICENSE file
    "security_txt": "/.well-known/security.txt", # GET: Standard security policy file

    "create_order": "/rest/orders/",
    "image_loader": "/images", # For image enumeration
    "file_download": "/download", # For file enumeration
      # Vulnerability Specific Endpoints (Common targets for challenges)
    "captcha_generation": "/api/Captcha/",      # GET: Generate CAPTCHA for feedback/registration
    "product_image_endpoint": "/rest/product/{product_id}/image", # GET: Product image, often vulnerable to XXE/SSRF
    "ftp_directory_listing": "/ftp/",           # GET: FTP directory listing (if misconfigured)
    "uploaded_complaints_directory": "/uploads/complaints/", # GET: Directory for uploaded complaints (LFI/RFI)
    "customer_feedback_frontend": "/contact",   # Frontend for feedback form, target for SSTI
    "admin_route_debug_endpoint": "/rest/admin/debug", # Hypothetical debug endpoint for admin routes (challenge)
    "product_file_upload": "/api/Products/{product_id}/upload", # Example product-related upload endpoint
    "b2b_sales_endpoint": "/rest/user/b2b", # A common challenge endpoint, POST: likely for B2B sales data upload
    "coupon_redemption": "/rest/coupon/apply/{coupon_code}", # POST or GET for coupon redemption
    "file_system_access_challenge": "/rest/admin/file-system", # Path traversal/LFI challenge
    "data_export_endpoint": "/rest/data/export", # For data export, can be vulnerable to LFI/SSRF

     "add_to_cart": "/api/BasketItems/", # Commonly /api/BasketItems/ in Juice Shop
        "password_reset": "/rest/user/forgot-password", # Specific to Juice Shop's reset process
        "admin_panel": "/rest/admin/users/{user_id}", # Hypothetical admin action on a user
        "update_product": "/api/Products/{product_id}", # Product update (requires admin/privileged user)
        "delete_product": "/api/Products/{product_id}", # Product deletion (requires admin/privileged user)
        "user_profile_update": "/api/Users/{user_id}", # Update user's own profile (PUT)
        "place_order": "/api/Orders/", # For checkout
        "address": "/api/Addresses/{address_id}", # For PUT/DELETE specific address
        "card": "/api/Cards/{card_id}", # For PUT/DELETE specific card
        "change_security_answer": "/rest/user/security-question", # For changing security answer
        "basket_item": "/api/BasketItems/{item_id}", # For PUT/DELETE specific basket item
        "data_export_endpoint": "/rest/data/export", # Reusing from DT, if it accepts POST/PUT
        

    }


   

    # --- Common Query Parameters ---
    COMMON_QUERY_PARAMS = ["q", "query", "id", "category", "search", "term", "file", "path", "data"]

    # Common usernames for enumeration attempts
    COMMON_USERNAMES = ["admin", "root", "test", "user", "guest", "administrator", "support"]

    # CSRF related configurations
    CSRF_TOKEN_NAMES = ["_csrf", "csrf_token", "authenticity_token"]
    CSRF_HEADERS = {"X-CSRF-TOKEN": "invalid_token"}


