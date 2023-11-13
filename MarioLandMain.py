# IMPORTS 
import multiprocessing
from shlex import join
from AgentManager import AgentManager
from MemoryManager import MemoryManager
from sshkeyboard import listen_keyboard
from VisualizationBoard import *
from utilities import *
import time
import gc
from DQNAgent import *
from pyboy import PyBoy, WindowEvent
import sys


CONTINUE = 1
NEW = 0

def main():

    
    #Path to the game 
    rom_path = "Super-Mario-Land-JUE-V1.1-.gb"

    #List of the possible action
    possible_actions = [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A]
    release = [WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_BUTTON_A]

    # GAME LAUNCHING 
    pyboy = PyBoy(rom_path, game_wrapper=True , disable_renderer=True)
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

    total_steps = 0
    reward_epi = 0
    fr = 0
    action_size = len(possible_actions)

    nbChan = 5
    #nbChan = 1

    
    agent_manager = AgentManager(action_size, nbChan, MODE)
    visualisation = VisualizationBoard("MarioLand_history.txt", disable_visualization = True)
    memory_manager = MemoryManager(threshold=95)
    total_episodes = visualisation.nb_episodes
    print("Begin at episode :",total_episodes)
    current_states = np.transpose(np.array(get_states(nbChan,15,mario,pyboy)), (1, 2, 0))
    last_mario = LastMario(mario)

    # Loop for the training session 
    episodesMax = total_episodes + 250
    frame = 0
    while total_episodes < episodesMax:


        action = possible_actions[agent_manager.train_agent.compute_action(current_states)]
        visualisation.show_action_bar(agent_manager.train_agent.q_values)
        
        pyboy.send_input(action)
        
        next_states = np.transpose(np.array(get_states(nbChan,15,mario,pyboy)), (1, 2, 0))

        pyboy.send_input(release[possible_actions.index(action)])

        frame += 1
        #reward = reward1(mario,last_lives,last_score)
        reward = reward_function(mario,last_mario)

        reward_epi += reward
        reward_epi = round(reward_epi,2)
        os.system('clear')
        print("\n\n\n\n\n")
        memory_manager.display_memory_info()
        print("Numbers of episodes",total_episodes,"| Exploration proba",agent_manager.train_agent.exploration_proba,
            "\nReward Action",reward,"| Total Reward Episode",reward_epi,"| Frame n°",frame)
        
        next_input = np.array(next_states)

        # We sotre each experience in the memory buffer
        agent_manager.train_agent.store_episode(current_states, possible_actions.index(action), reward, next_input)
        current_states = next_states
        last_mario.updateLastMario(mario)

        if mario.lives_left==0 or mario.time_left<200:
            print("End of the current episode")
            if agent_manager.train_agent.experience_memory.get_size() >= agent_manager.train_agent.train_size:
                total_steps = 0
                history = agent_manager.train_agent.train(agent_manager.target_agent)
                history["exploration_proba"] = agent_manager.train_agent.exploration_proba
                history["reward_episode"] = reward_epi
                total_episodes+=1

                reward_epi = 0
                with open("MarioLand_history.txt", 'a') as file:
                    json.dump(history, file)
                    file.write('\n')
                
                agent_manager.train_agent.update_exploration_probability()
            print("Updating agent exploration_proba to",agent_manager.train_agent.exploration_proba)

            #Updating plot
            visualisation.save_rewards_plots(save_path="reward_plot.png")
            visualisation.save_loss_and_accuracy_plots(save_path="loss_accuracy_plot.png")
            visualisation.last_loss_and_accuracy_plots(save_path="last_loss_accuracy_plot.png")


            
            
            last_fitness=0
            
            mario.reset_game()
            frame = 0
            current_states = np.transpose(np.array(get_states(nbChan,15,mario,pyboy)), (1, 2, 0))
            last_mario.updateLastMario(mario)
            assert mario.lives_left == 2

            if total_episodes%50 == 0:
                agent_manager.target_agent.model.set_weights(agent_manager.train_agent.model.get_weights())
                agent_manager.target_agent.model.save('lastTarget_agentVersion.keras')
            
            agent_manager.train_agent.model.save('lastTrain_agentVersion.keras')
            save_buffer("saved_buffer.txt",agent_manager.train_agent)
            tf.keras.backend.clear_session()
            del agent_manager.train_agent.model
            del agent_manager.target_agent.model
            gc.collect(generation=2)
            print("CLEARING MEMORY")
            
            time.sleep(2) 
            agent_manager.target_agent.model = keras.models.load_model('lastTarget_agentVersion.keras')
            agent_manager.train_agent.model = keras.models.load_model('lastTrain_agentVersion.keras')
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
            print("Training paused. Press 'r' to resume.")
        elif key == 'r':
            pause_event.set()
            print("Training resumed.")
        elif key == 'q':
            print("Training exit.")
            process.kill()
            sys.exit()

    def release(key):
        pass  # You can add logic for key release if needed

    listener = listen_keyboard(on_press=press, on_release=release)


if __name__ == "__main__":
    main()
    
    """pause_event = multiprocessing.Event()
    started_event = multiprocessing.Event()  # to know when the child started execution
    # consider making this next process daemon for guaranteed cleanup
    p = multiprocessing.Process(target=main, args=(pause_event, started_event))
    p.daemon=True
    p.start()
    started_event.wait(timeout=5)  # to fix window's buggy terminal
    control_process(p, pause_event)
    """
