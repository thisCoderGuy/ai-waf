import random
from locust import HttpUser, TaskSet, task, between

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
    wait_time = between(1, 5)

    host = AppConfig.BASE_HOST

    tasks = {
        LegitimateUser: AppConfig.USER_TASK_WEIGHTS["LegitimateUser"],
        SQLiAttacker: AppConfig.USER_TASK_WEIGHTS["SQLiAttacker"],
        XSSAttacker: AppConfig.USER_TASK_WEIGHTS["XSSAttacker"],
        DirectoryTraversalAttacker: AppConfig.USER_TASK_WEIGHTS["DirectoryTraversalAttacker"],
        EnumerationAttacker: AppConfig.USER_TASK_WEIGHTS["EnumerationAttacker"],
        CSRFAttacker: AppConfig.USER_TASK_WEIGHTS["CSRFAttacker"],
    }

    def on_start(self):
        """
        Called when a Locust user starts. Sets a random legitimate User-Agent.
        """
        self.client.headers["User-Agent"] = random.choice(AppConfig.USER_AGENTS)