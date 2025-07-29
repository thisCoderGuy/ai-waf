import random
import time
import os
from locust import HttpUser, TaskSet, task, between, events
from locust.event import EventHook
from locust.env import Environment
from locust.runners import MasterRunner, LocalRunner 



from locust_tests.config import AppConfig
from locust_tests.legitimate_user import LegitimateUser
from locust_tests.attackers.sqli import SQLiAttacker
from locust_tests.attackers.xss import XSSAttacker
from locust_tests.attackers.directory_traversal import DirectoryTraversalAttacker
from locust_tests.attackers.enumeration import EnumerationAttacker
from locust_tests.attackers.csrf import CSRFAttacker

# --- Global State for Phase Synchronization ---
# This variable is updated by the master/phase controller via events.
_current_campaign_phase = None
_campaign_finished = False 

# --- Centralized Traffic Label Management ---
last_written_label = None 

def set_traffic_label_safely(verdict: str, traffic_type: str):
    """
    Writes the current traffic label to the shared file.
    Only writes if the label is new or different from the last one written.
    This function should ideally only be called by the phase controller.
    """
    global last_written_label
    combined_label = f"{verdict.upper()}_{traffic_type.upper()}"

    if combined_label != last_written_label:
        try:
            os.makedirs(os.path.dirname(AppConfig.SHARED_CONFIG_FILE_PATH), exist_ok=True)
            with open(AppConfig.SHARED_CONFIG_FILE_PATH, "w") as f:
                f.write(combined_label)
            print(f"[{time.time()}] Master: Set traffic label to: {combined_label}")
            last_written_label = combined_label
        except Exception as e:
            print(f"[{time.time()}] Master: Error writing to shared config file '{AppConfig.SHARED_CONFIG_FILE_PATH}': {e}")

# --- Custom Locust Event for Phase Changes ---
# This allows the master to broadcast phase changes to all workers/users.
# 'campaign_phase_event' is a custom event name.
campaign_phase_event = EventHook()

@campaign_phase_event.add_listener
def _on_campaign_phase_change(environment, phase_info, **kwargs):
    """
    Listener that updates the global current phase on all workers/users.
    """
    global _current_campaign_phase
    global _campaign_finished

    phase_name = phase_info.get("name")
    action = phase_info.get("action")

    if action == "start_phase":
        _current_campaign_phase = phase_name
        print(f"[{time.time()}] ################################### User/Worker: Global phase changed to: {_current_campaign_phase}")
    elif action == "end_campaign":
        _campaign_finished = True
        _current_campaign_phase = "CAMPAIGN_END"
        print(f"[{time.time()}] User/Worker: Global campaign finished.")

# --- Main TaskSet for Users ---
class AttackCampaign(TaskSet):
    """
    This TaskSet represents the user's role in the campaign.
    Each user continuously checks the global phase and executes tasks for that phase.
    """
    wait_time = between(0.1, 0.5) # Short wait to quickly react to phase changes

    def on_start(self):
        print(f"[{time.time()}] User: Starting to monitor campaign phases...")

    @task
    def run_current_phase_tasks(self):
        global _current_campaign_phase
        global _campaign_finished

        if _campaign_finished:
            print(f"[{time.time()}] User: Campaign finished, stopping tasks.")
            self.user.environment.runner.quit() 
            return

        current_phase_name = _current_campaign_phase

        if current_phase_name is None:
            time.sleep(1) 
            return

        phase_map = {
            "BENIGN_BENIGN": LegitimateUser,
            "MALICIOUS_SQLI": SQLiAttacker,
            "MALICIOUS_XSS": XSSAttacker,
            "MALICIOUS_DT": DirectoryTraversalAttacker,
            "MALICIOUS_CSRF": CSRFAttacker,
            "MALICIOUS_ENUM": EnumerationAttacker,
        }

        target_task_set_class = phase_map.get(current_phase_name)

        if target_task_set_class:
            sub_task_set = target_task_set_class(self.user)
            sub_task_set.on_start() 

            # We don't use a fixed duration here, instead we run a few tasks
            # and then check the global phase again.
            tasks_to_run_per_check = 5 

            for _ in range(tasks_to_run_per_check):
                if _current_campaign_phase != current_phase_name or _campaign_finished:
                    break

                try:
                    chosen_task_func = sub_task_set.get_next_task()
                    if chosen_task_func:
                        chosen_task_func(sub_task_set)
                        time.sleep(sub_task_set.wait_time()) 
                    else:
                        print(f"[{time.time()}] WARNING: No callable tasks found in sub-TaskSet {target_task_set_class.__name__}")
                        break
                except Exception as e:
                    print(f"[{time.time()}] Error in sub-TaskSet {target_task_set_class.__name__}: {e}")
                    # Allow to continue to next check to see if phase changed
                    break
            
            sub_task_set.on_stop() 
        else:
            print(f"[{time.time()}] User: Waiting for phase: {current_phase_name}")
            time.sleep(1) 

# --- User Class Definition ---
class WebsiteUser(HttpUser):
    host = "http://coraza-proxy:8080"
    wait_time = between(1, 3) # This wait_time now mostly applies when switching TaskSets
                              # The AttackCampaign's wait_time and sub-taskset's wait_time are more relevant
    tasks = [AttackCampaign]


# --- Global Phase Controller (Activated on Master/Single Locust Instance) ---
class PhaseController:
    """
    Manages the overall campaign phases and broadcasts changes.
    This class is instantiated and run only on the master process.
    """
    PHASES = [
        {"name": "BENIGN_BENIGN", "duration_key": "LegitimateUser"},
        {"name": "MALICIOUS_SQLI", "duration_key": "SQLiAttacker"},
        {"name": "MALICIOUS_XSS", "duration_key": "XSSAttacker"},
        {"name": "MALICIOUS_DT", "duration_key": "DirectoryTraversalAttacker"},
        {"name": "MALICIOUS_CSRF", "duration_key": "CSRFAttacker"},
        {"name": "MALICIOUS_ENUM", "duration_key": "EnumerationAttacker"},
    ]

    def __init__(self, environment):
        self.environment = environment
        self.running = True

    def start_campaign(self):
        """Orchestrates the campaign phases."""
        print(f"[{time.time()}] Master: Campaign Controller Starting...")
        set_traffic_label_safely("CAMPAIGN", "START")
        
        # Broadcast initial state
        campaign_phase_event.fire(environment=self.environment, phase_info={"name": "CAMPAIGN_START", "action": "start_phase"})
        
        for phase_config in self.PHASES:
            phase_name = phase_config["name"]
            duration_key = phase_config["duration_key"]
            duration_seconds = AppConfig.PHASE_LENGTHS_SECONDS.get(duration_key, 60) # Default to 60s if not found

            if duration_seconds <= 0:
                print(f"[{time.time()}] ################### Master: Skipping phase: {phase_name} as duration is {duration_seconds} seconds.")
                continue 

            print(f"[{time.time()}] ################### Master: Initiating phase: {phase_name} for {duration_seconds} seconds")
            
            # Update and broadcast the phase to all users/workers
            set_traffic_label_safely(phase_name.split('_')[0], phase_name.split('_')[1])
            campaign_phase_event.fire(environment=self.environment, phase_info={"name": phase_name, "action": "start_phase"})

            # Wait for the duration of this phase
            time.sleep(duration_seconds)
            time.sleep(2) # Short buffer after phase for printout/transition

            print(f"[{time.time()}] Master: Phase {phase_name} completed.")

        print(f"[{time.time()}] Master: All campaign phases completed. Signalling end.")
        set_traffic_label_safely("CAMPAIGN", "END")
        campaign_phase_event.fire(environment=self.environment, phase_info={"name": "CAMPAIGN_END", "action": "end_campaign"})
        
        # Give a moment for events to propagate and users to react
        time.sleep(5)
       
        if self.environment.runner:
            print(f"[{time.time()}] Master: Quitting Locust runner.")
            self.environment.runner.quit()


# --- Event Hook to Start the Phase Controller ---
def on_test_start_hook(environment: Environment, **kwargs):
    """
    Called once when the test starts.
    We only want the master to run the phase controller.
    """
    # Initialize the controller only if it's the master or running in standalone mode
    if isinstance(environment.runner, MasterRunner) or isinstance(environment.runner, LocalRunner):
        print(f"[{time.time()}] Initializing PhaseController on {type(environment.runner).__name__}.")
        controller = PhaseController(environment)
        import threading
        campaign_thread = threading.Thread(target=controller.start_campaign)
        campaign_thread.daemon = True
        campaign_thread.start()


# Register the event hook
events.test_start.add_listener(on_test_start_hook)

# Optional: Ensure cleanup of the label file on test stop
@events.test_stop.add_listener
def on_test_stop_hook(environment: Environment, **kwargs):
    if environment.runner and environment.runner.is_master:
        print(f"[{time.time()}] Master: Test stopped. Cleaning up label file (optional).")
        try:
            if os.path.exists(AppConfig.SHARED_CONFIG_FILE_PATH):
                # os.remove(AppConfig.SHARED_CONFIG_FILE_PATH) 
                pass 
        except Exception as e:
            print(f"[{time.time()}] Master: Error during cleanup: {e}")