import psutil
import sys

class MemoryManager:
    def __init__(self, threshold=95):
        self.threshold = threshold

    def get_memory_usage_percentage(self):
        # Get the current memory usage in percentage
        memory_percent = psutil.virtual_memory().percent
        return memory_percent

    def check_memory_usage(self):
        memory_percent = self.get_memory_usage_percentage()

        print(f"Memory Usage: {memory_percent}%")

        if memory_percent > self.threshold:
            print("Memory usage exceeds the threshold. Stopping the script.")
            sys.exit()

    def display_memory_info(self):
        # Display detailed memory information
        memory_info = psutil.virtual_memory()
        print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
        print(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB")
        print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
        print(f"Memory Usage Percentage: {memory_info.percent}%")

    def get_available_memory(self):
        # Get the available memory in GB
        memory_info = psutil.virtual_memory()
        return memory_info.available / (1024 ** 3)

if __name__ == "__main__":
    # Example usage when the script is run as the main program
    memory_manager = MemoryManager(threshold=95)
    memory_manager.check_memory_usage()
    memory_manager.display_memory_info()
    available_memory = memory_manager.get_available_memory()
    print(f"Available Memory: {available_memory:.2f} GB")