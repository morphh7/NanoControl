import os
import time

class path:
    def __init__(self):
        self.main_dir: str = self.find_main_path()
        self.temp_dir: str = self.find_and_create_temp()

    def find_main_path(self) -> str:
        """
        Returns the absolute path of the directory containing this file.

        - params: none
        - return: main folder path
        """
        return os.path.dirname(os.path.abspath(__file__))

    def find_and_create_temp(self) -> str:
        """
        Finds or creates the temp/dump folder based on environment vars or local path.

        - params: none
        - return: temp folder path
        """
        path = os.environ.get(
            "NANOCONTROL_TEMP",
            os.path.join(self.find_main_path(), "..", "temp"),
        )
        os.makedirs(path, exist_ok=True)
        return os.path.abspath(path)

class timer:
    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0

    def start(self):
        """starts the timer"""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """
        stops the timer
        
        - params: none
        - return: time elapsed since starting
        """

        self.end_time = time.perf_counter()
        return (self.end_time - self.start_time)

    @property
    def elapsed(self) -> float:
        """calculates the elapsed time without stopping"""
        return (time.perf_counter() - self.start_time)