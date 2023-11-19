import os
import json
import numpy as np
import time
import copy

class LastMario():
    def __init__(self, mario):
        self.updateLastMario(mario)
    
    def updateLastMario(self,mario):
        self.world = mario.world
        self.coins = mario.coins
        self.lives_left = mario.lives_left
        self.score = mario.score
        self.time_left = mario.time_left 
        self.level_progress = mario.level_progress 
        self._level_progress_max = mario._level_progress_max 
        self.fitness = mario.fitness 

    def display_info(self):
        print("World:", self.world)
        print("Coins:", self.coins)
        print("Lives Left:", self.lives_left)
        print("Score:", self.score)
        print("Time Left:", self.time_left)
        print("Level Progress:", self.level_progress)
        print("Max Level Progress:", self._level_progress_max)
        print("Fitness:", self.fitness)

def get_states(nbChannel,nbFrame,mario,pyboy):
    states = []
    for state in range(nbFrame): # (action duration = freeze mario)
        pyboy.tick()
        if state  % 3 == 0 and nbChannel > 1:
            states.append(mario.game_area())
        elif state == 14 and nbChannel == 1:
            states.append(mario.game_area())
    return states

def remove_file(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
    # Remove the file
        os.remove(file_path)
    return

def save_buffer(file_path,agent):
    with open(file_path, "w") as file:
        for entry in agent.experience_memory.memory_buffer:
            entry_to_save = {
                "current_state": entry["current_state"].tolist(),
                "action": entry["action"],
                "reward": entry["reward"],
                "next_state": entry["next_state"].tolist()
            }
            json_entry = json.dumps(entry_to_save)  # Convert the dictionary to a JSON string
            file.write(json_entry + "\n")  # Write the JSON string to the file

def load_buffer(file_path,agent):
    # Open the file in read mode and load the data
    with open(file_path, "r") as file:
        for line in file:
            # Parse the JSON-formatted string into a dictionary
            entry = json.loads(line)
            entry["current_state"] = np.array(entry["current_state"])
            entry["next_state"] = np.array(entry["next_state"])
            agent.experience_memory.add_experience(entry)


def reward1(mario,last_lives,last_score):
    if mario.lives_left<last_lives:
        reward = mario.score-last_score-10000+(mario.time_left*15)
    else:
        reward = mario.score-last_score+(mario.time_left*5)
    reward_progress = mario.level_progress*mario.time_left/100
    print("Reward score :",reward,"| Reward progress :",reward_progress,"| Sum reward :",reward+reward_progress,"| Max progress :",mario._level_progress_max)
    reward += reward_progress
    return reward

def reward2(mario,last_lives,last_score,last_progress):
    reward = (mario.score-last_score)*(mario.time_left/100)
    reward_progress = max(-500,(mario.level_progress-last_progress-10)*mario.time_left/10)
    print("Reward score :",reward,"| Reward progress :",reward_progress,"| Sum reward :",reward+reward_progress,"| Max progress :",mario._level_progress_max)
    reward += reward_progress
    return reward


def reward_function(mario,last_mario):
    # Define initial reward
    reward = 0
    # Check if the episode is done (Mario won or lost)
    if mario.world[0]>last_mario.world[0] or mario.world[1]>last_mario.world[1]:
        reward += 100
    # Check for positive rewards
    if mario.coins > last_mario.coins:
        reward += 5  # Reward for collecting a coin
    if mario.score > last_mario.score:
        reward += 20  # Reward for increasing the score
    if mario.lives_left < last_mario.lives_left:
        reward -= 20
    # Check for negative rewards
    if mario.level_progress < last_mario.level_progress:
        reward -= 2  # Penalty for moving backward
    if mario._level_progress_max > last_mario._level_progress_max:
        reward += 5  # Penalty for moving backward
    elif mario.level_progress > last_mario.level_progress:
        reward += 3  # Reward for moving forward
    else:
        reward -= 1  # Slight penalty for staying in the same position
    reward = round(reward, 2)
    return reward