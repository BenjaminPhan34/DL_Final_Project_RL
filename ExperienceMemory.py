import numpy as np

class ExperienceMemory:
    def __init__(self, max_size=10000):
        self.memory_buffer = []
        self.max_size = max_size

    def add_experience(self, experience):
        self.memory_buffer.append(experience)
        # If the size of the memory buffer exceeds its maximum, remove the oldest experience
        if len(self.memory_buffer) > self.max_size:
            del self.memory_buffer[0]

    def get_batch_sample(self, batch_size):
        memory_buffer_tmp = self.memory_buffer.copy()
        np.random.shuffle(memory_buffer_tmp)
        batch_sample = memory_buffer_tmp[:batch_size]
        del memory_buffer_tmp
        return batch_sample

    def clear_memory(self):
        del self.memory_buffer
        self.memory_buffer = []