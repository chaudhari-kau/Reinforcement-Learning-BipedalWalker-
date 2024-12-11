import time
from datetime import datetime, timedelta

class TimeEstimator:
    def __init__(self, total_iterations, name="Process"):
        self.start_time = time.time()
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.name = name

    def update(self, iteration):
        self.current_iteration = iteration
        elapsed_time = time.time() - self.start_time
        progress = self.current_iteration / self.total_iterations
        
        if progress > 0:
            estimated_total = elapsed_time / progress
            remaining_time = estimated_total - elapsed_time
            
            print(f"\r{self.name}: {progress*100:.1f}% complete | "
                  f"Elapsed: {str(timedelta(seconds=int(elapsed_time)))} | "
                  f"Remaining: {str(timedelta(seconds=int(remaining_time)))} | "
                  f"ETA: {datetime.now() + timedelta(seconds=int(remaining_time))}", 
                  end="")

    def finish(self):
        total_time = time.time() - self.start_time
        print(f"\n{self.name} completed in {str(timedelta(seconds=int(total_time)))}")