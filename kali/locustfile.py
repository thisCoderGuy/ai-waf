import random
from locust import HttpUser, between

# ALL IMPORTS MUST BE ABSOLUTE NOW, STARTING FROM THE PACKAGE NAME 'locust_tests'
from locust_tests.config import AppConfig
from locust_tests.legitimate_user import LegitimateUser
from locust_tests.attackers.sqli import SQLiAttacker
from locust_tests.attackers.xss import XSSAttacker
from locust_tests.attackers.directory_traversal import DirectoryTraversalAttacker
from locust_tests.attackers.enumeration import EnumerationAttacker
from locust_tests.attackers.csrf import CSRFAttacker


class WebsiteUser(HttpUser):
    """
    Main Locust user class that orchestrates the behavior of different user types.
    """
    # wait_time defines the simulated user's idle time between tasks.
    # between(min_wait, max_wait) means users will wait a random time
    # between min_wait and max_wait seconds.
    wait_time = between(1, 5)

    host = AppConfig.BASE_HOST

    # A task refers to a specific behavior or action that a simulated user will perform.
    # Instead of listing individual task methods with @task decorators directly within WebsiteUser, 
    # this dictionary assigns weights to other TaskSet classes.
    tasks = {
        LegitimateUser: AppConfig.USER_TASK_WEIGHTS["LegitimateUser"], # from locust_tests
        SQLiAttacker: AppConfig.USER_TASK_WEIGHTS["SQLiAttacker"], # from locust_tests.attackers
        XSSAttacker: AppConfig.USER_TASK_WEIGHTS["XSSAttacker"], # from locust_tests.attackers
        DirectoryTraversalAttacker: AppConfig.USER_TASK_WEIGHTS["DirectoryTraversalAttacker"], # from locust_tests.attackers
        EnumerationAttacker: AppConfig.USER_TASK_WEIGHTS["EnumerationAttacker"], # from locust_tests.attackers
        CSRFAttacker: AppConfig.USER_TASK_WEIGHTS["CSRFAttacker"], # from locust_tests.attackers
    }

    def on_start(self):
        """
        Called when a Locust user starts. Sets a random legitimate User-Agent.
        """
        self.client.headers["User-Agent"] = random.choice(AppConfig.USER_AGENTS)