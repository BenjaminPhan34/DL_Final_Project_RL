from DQNAgent import DQNAgent
from sshkeyboard import listen_keyboard
import keras
from utilities import *
import time

CONTINUE = True
NEW = False

class AgentManager:

    def __init__(self, action_size, nb_channels, state_training = NEW):
        # Create DQNAgent instances for training and target
        self.train_agent = DQNAgent(action_size, nb_channels)
        self.target_agent = DQNAgent(action_size, nb_channels)

        # Set the training mode
        self.MODE = state_training
        if self.MODE == CONTINUE:
            self.resume_training()
        elif self.MODE == NEW:
            self.new_training_session()


        # Flag to indicate whether training is paused
        self.training_paused = False

    def resume_training(self):
        """
        Resume training from a previous session.

        Parameters:
        - train_model_path (str): File path to the saved training model.
        - target_model_path (str, optional): File path to the saved target model.
        - buffer_path (str, optional): File path to the saved memory buffer.

        Returns:
        None
        """
        if self.target_agent:
            self.target_agent.model = keras.models.load_model('lastTarget_agentVersion.keras')
        self.train_agent.model = keras.models.load_model('lastTrain_agentVersion.keras')
        with open("MarioLand_history.txt", 'r') as file:
            data = [json.loads(line) for line in file]
            self.train_agent.exploration_proba = data[-1]["exploration_proba"]
        if os.path.exists("saved_buffer.txt"):
            load_buffer("saved_buffer.txt",self.train_agent)


    def new_training_session(self):
        #Clear previous training session if necessary 
        remove_file("saved_buffer.txt")
        remove_file("MarioLand_history.txt")
        f = open("MarioLand_history.txt", 'a')
        f.close()
        f = open("saved_buffer.txt", 'a')
        f.close()
        self.target_agent.model.save('lastTarget_agentVersion.keras')