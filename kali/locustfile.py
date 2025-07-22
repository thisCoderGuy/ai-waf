import random
from locust import HttpUser, between

# ALL IMPORTS MUST BE ABSOLUTE NOW, STARTING FROM THE PACKAGE NAME 'locust_tests'
from locust_tests.config import AppConfig
from locust_tests.attack_campaign import AttackCampaign



class WebsiteUser(HttpUser):
    """
    Main Locust user class that orchestrates the behavior of different user types.
    """
    # wait_time defines the simulated user's idle time between tasks.
    # between(min_wait, max_wait) means users will wait a random time
    # between min_wait and max_wait seconds.
    wait_time = between(0.001, 0.005)

    host = AppConfig.BASE_HOST

    # A task refers to a specific behavior or action that a simulated user will perform.

    
    # Sequential tasks
    tasks = [AttackCampaign]
    