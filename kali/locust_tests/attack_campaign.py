
import random
import time
from locust import TaskSet, HttpUser, task, between, SequentialTaskSet

from .config import AppConfig

from locust_tests.legitimate_user import LegitimateUser
from locust_tests.attackers.sqli import SQLiAttacker
from locust_tests.attackers.xss import XSSAttacker
from locust_tests.attackers.directory_traversal import DirectoryTraversalAttacker
from locust_tests.attackers.enumeration import EnumerationAttacker
from locust_tests.attackers.csrf import CSRFAttacker



def set_traffic_label(verdict: str, type: str):
    """Writes the current traffic label to the shared file."""
    try:
        combined_label = f"{verdict}_{type}"
        with open(AppConfig.SHARED_CONFIG_FILE_PATH, "w") as f:
            f.write(combined_label)
        print(f"[{time.time()}] Set traffic label to: {combined_label}")
    except Exception as e:
        print(f"[{time.time()}] Error writing to shared config file: {e}")


class AttackCampaign(SequentialTaskSet):    
    """
    This TaskSet orchestrates the entire attack campaign, running
    different types of attacks in a predefined sequence.
    """
    def on_start(self):
        print(f"[{time.time()}] Starting Attack Campaign...")

    @task
    def run_benign_phase(self):
        print(f"[{time.time()}] Starting Benign Traffic Phase...")
        set_traffic_label("Benign", "Benign")

        # Start a sub-TaskSet for benign traffic
        self.run_sub_taskset(LegitimateUser, duration_seconds=AppConfig.PHASE_LENGTHS_SECONDS["LegitimateUser"]) # Run legitimate traffic for configured seconds
        time.sleep(5) 
        print(f"[{time.time()}] Benign Traffic Phase Completed.")

    @task
    def run_sqli_phase(self):
        print(f"[{time.time()}] Starting SQLi Attack Phase...")
        set_traffic_label("Malicious", "SQLi")

        self.run_sub_taskset(SQLiAttacker, duration_seconds=AppConfig.PHASE_LENGTHS_SECONDS["SQLiAttacker"]) # Run SQLi for configured seconds
        time.sleep(5)
        print(f"[{time.time()}] SQLi Attack Phase Completed.")

    @task
    def run_xss_phase(self):
        print(f"[{time.time()}] Starting XSS Attack Phase...")
        set_traffic_label("Malicious", "XSS")

        self.run_sub_taskset(XSSAttacker, duration_seconds=AppConfig.PHASE_LENGTHS_SECONDS["XSSAttacker"]) # Run XSS for configured seconds
        time.sleep(5)
        print(f"[{time.time()}] XSS Attack Phase Completed.")

    @task
    def run_directory_traversal_phase(self):
        print(f"[{time.time()}] Starting Directory Traversal Attack Phase...")
        set_traffic_label("Malicious", "DT")

        self.run_sub_taskset(DirectoryTraversalAttacker, duration_seconds=AppConfig.PHASE_LENGTHS_SECONDS["DirectoryTraversalAttacker"]) # Run  DT for configured seconds
        time.sleep(5)
        print(f"[{time.time()}] Directory Traversal Attack Phase Completed.")

    @task
    def run_csrf_phase(self):
        print(f"[{time.time()}] Starting CSRF Attack Phase...")
        set_traffic_label("Malicious", "CSRF")

        self.run_sub_taskset(CSRFAttacker, duration_seconds=AppConfig.PHASE_LENGTHS_SECONDS["CSRFAttacker"]) # Run CSRF for configured seconds
        time.sleep(5)
        print(f"[{time.time()}] CSRF Attack Phase Completed.")

    @task
    def run_enumeration_phase(self):
        print(f"[{time.time()}] Starting Enumeration Attack Phase...")
        set_traffic_label("Malicious", "Enum")

        self.run_sub_taskset(EnumerationAttacker, duration_seconds=AppConfig.PHASE_LENGTHS_SECONDS["EnumerationAttacker"]) # Run  Enumeration for configured seconds
        time.sleep(5)
        print(f"[{time.time()}] Enumeration Attack Phase Completed.")

    @task
    def end_campaign(self):
        print(f"[{time.time()}] Attack Campaign Finished.")
        set_traffic_label("CAMPAIGN", "END")
        self.user.environment.runner.quit() # Stop the Locust test after campaign finishes

    # Helper method to run a sub-TaskSet for a specified duration
    def run_sub_taskset(self, task_set_class, duration_seconds):
        start_time = time.time()
        sub_task_set = task_set_class(self.user)
        sub_task_set.on_start()

        while (time.time() - start_time) < duration_seconds:
            try:
                chosen_task_func = sub_task_set.get_next_task()
                if chosen_task_func is None:
                    print(f"[{time.time()}] WARNING: No callable tasks found in sub-TaskSet {task_set_class.__name__}")
                    break # Exit loop if no tasks to choose from

                chosen_task_func(sub_task_set)
                time.sleep(sub_task_set.wait_time())
            except Exception as e:
                print(f"[{time.time()}] Error in sub-TaskSet {task_set_class.__name__}: {e}")
                break # Break to avoid infinite loop on persistent errors


        sub_task_set.on_stop() # Call on_stop if it has any cleanup


           