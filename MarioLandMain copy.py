# IMPORTS 
import multiprocessing
from AgentManager import AgentManager
from MemoryManager import MemoryManager
from sshkeyboard import listen_keyboard
from VisualizationBoard import *
from utilities import *
import time
import gc
from DQNAgent import *
from pyboy import PyBoy, WindowEvent

CONTINUE = 1
NEW = 0

#Path to the game 
rom_path = "Super-Mario-Land-JUE-V1.1-.gb"

#List of the possible action
possible_actions = [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A]
release = [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A]

# GAME LAUNCHING 
pyboy = PyBoy(rom_path, game_wrapper=True)
pyboy.set_emulation_speed(0) #cocaine mode  (ATTENTION: skip every pyboy tick)
assert pyboy.cartridge_title() == "SUPER MARIOLAN"
mario = pyboy.game_wrapper()
mario.start_game(timer_div=None, world_level=None, unlock_level_select=False)

assert mario.score == 0
assert mario.lives_left == 2
assert mario.time_left == 400
assert mario.world == (1, 1)
assert mario._level_progress_max == 0
assert mario.fitness == 0 # A built-in fitness score for AI development

###### IMPORTANT CHOOSE THE MODE "NEW" TRAINING OR "CONTINUE" TRAINING ######
MODE = CONTINUE

###### Power off computer if True
AUTOSHUTDOWN = False

#Initialisation
last_fitness = 0
total_episodes = 1
total_steps = 0
reward_epi = 0
fr = 0
action_size = len(possible_actions)

nbChan = 5
#nbChan = 1

train_agent = DQNAgent(action_size,nbChan) 
target_agent = DQNAgent(action_size,nbChan) 


# Resume the training of the last model trained
if MODE == CONTINUE:
    if target_agent:
        target_agent.model = keras.models.load_model('lastTarget_agentVersion.keras')
    train_agent.model = keras.models.load_model('lastTrain_agentVersion.keras')
    with open("MarioLand_history.txt", 'r') as file:
        data = [json.loads(line) for line in file]
        train_agent.exploration_proba = data[-1]["exploration_proba"]
    if os.path.exists("saved_buffer.txt"):
        load_buffer("saved_buffer.txt",train_agent)
# Create a training session
elif MODE == NEW:
    #Clear previous training session if necessary 
    remove_file("saved_buffer.txt")
    remove_file("MarioLand_history.txt")
    f = open("MarioLand_history.txt", 'a')
    f.close()
    f = open("saved_buffer.txt", 'a')
    f.close()
    target_agent.model.save('lastTarget_agentVersion.keras')



def main(pause_event, event_started):
    event_started.set()
    agent_manager = AgentManager(action_size, nbChan, MODE)
    visualisation = VisualizationBoard("MarioLand_history.txt")
    memory_manager = MemoryManager(threshold=95)

    current_states = np.transpose(np.array(get_states(nbChan,15,mario,pyboy)), (1, 2, 0))
    last_mario = LastMario(mario)
    pause_episode = 100
    # Loop for the training session 
    episodesMax = 500
    while total_episodes < episodesMax and not agent_manager.training_paused:
    
        
        if fr == 1: #fps  (decision-taking frequence)
            fr = 0

            action = possible_actions[agent_manager.train_agent.compute_action(current_states)]
            visualisation.show_action_bar(agent_manager.train_agent.q_values)
            
            pyboy.send_input(action)
            
            next_states = np.transpose(np.array(get_states(nbChan,15,mario,pyboy)), (1, 2, 0))

            pyboy.send_input(release[possible_actions.index(action)])

            #reward = reward1(mario,last_lives,last_score)
            reward = reward_function(mario,last_mario)

            reward_epi += reward
            reward_epi = round(reward_epi,2)
            os.system('clear')
            print("Numbers of episodes",total_episodes,"| Exploration proba",agent_manager.train_agent.exploration_proba,
                "\nReward Action",reward,"| Total Reward Episode",reward_epi)
            
            next_input = np.array(next_states)

            # We sotre each experience in the memory buffer
            train_agent.store_episode(current_states, possible_actions.index(action), reward, next_input)
            current_states = next_states
            last_mario.updateLastMario(mario)

        if mario.lives_left==0 or mario.time_left<200:
            print("End of the current episode")
            if train_agent.experience_memory.max_size >= train_agent.train_size:
                total_steps = 0
                history = train_agent.train(target_agent)
                history["exploration_proba"] = train_agent.exploration_proba
                history["reward_episode"] = reward_epi
                total_episodes+=1

                reward_epi = 0
                with open("MarioLand_history.txt", 'a') as file:
                    json.dump(history, file)
                    file.write('\n')

                #Updating plot
                visualisation.save_rewards_plots(save_path="reward_plot.png")
                visualisation.save_loss_and_accuracy_plots(save_path="loss_accuracy_plot.png")


            train_agent.update_exploration_probability()
            print("Updating agent exploration_proba to",train_agent.exploration_proba)
            
            last_fitness=0
            
            mario.reset_game()
            current_states = np.transpose(np.array(get_states(nbChan,15,mario,pyboy)), (1, 2, 0))
            last_mario.updateLastMario(mario)
            assert mario.lives_left == 2

            if total_episodes%50 == 0:
                target_agent.model.set_weights(train_agent.model.get_weights())
                target_agent.model.save('lastTarget_agentVersion.keras')
            
            train_agent.model.save('lastTrain_agentVersion.keras')
            save_buffer("saved_buffer.txt",train_agent)
            tf.keras.backend.clear_session()
            del train_agent.model
            del target_agent.model
            gc.collect(generation=2)
            print("CLEARING MEMORY")
            memory_manager.display_memory_info()
            time.sleep(2) 
            target_agent.model = keras.models.load_model('lastTarget_agentVersion.keras')
            train_agent.model = keras.models.load_model('lastTrain_agentVersion.keras')
    # Save the model
    pyboy.stop() 

    if AUTOSHUTDOWN:
        os.system('shutdown.exe /s /t 0')

def control_process(process, pause_event):
    """
    Receive a process object as an argument and provides the ability
    to pause and resume the process.
    """
    def press(key):
        if key == 'p':
            pause_event.clear()
            print("Training paused. Press 'p' to resume.")
        elif key == 'r':
            pause_event.set()
            print("Training resumed.")
        elif key == 'q':
            print("Training exit.")
            process.terminate()


    def release(key):
        pass  # You can add logic for key release if needed

    listener = listen_keyboard(on_press=press, on_release=release)


if __name__ == "__main__":
    pause_event = multiprocessing.Event()
    started_event = multiprocessing.Event()  # to know when the child started execution
    # consider making this next process daemon for guaranteed cleanup
    p = multiprocessing.Process(target=main, args=(pause_event, started_event))
    p.start()

    started_event.wait(timeout=5)  # to fix window's buggy terminal
    control_process(p, pause_event)