import numpy as np
import tensorflow as tf
from ExperienceMemory import ExperienceMemory
from tensorflow import keras
from tensorflow.keras import layers, models
import os


class DQNAgent:
    def __init__(self, action_size, nbChannel, LR=0.0001, gamma=0.99, exploration_proba=1, exploration_proba_decay=0.005,
                 batch_size=64, train_size=2560):
        self.action_size = action_size
        self.lr = LR
        self.gamma = gamma
        self.exploration_proba = exploration_proba
        self.exploration_proba_decay = exploration_proba_decay
        self.batch_size = batch_size
        self.train_size = train_size
        self.nbChannel = nbChannel

        self.experience_memory = ExperienceMemory()

        self.Model3()  # Default to Model2, you can switch to Model1 if needed
        self.model.summary()

        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr),
                           loss=keras.losses.Huber(),
                           metrics=['accuracy'])

    def compute_action(self, current_state):
        """
        Compute the "best" action to perform given a state.
        """
        current_state = np.expand_dims(current_state, axis=0)

        if np.random.uniform(0, 1) < self.exploration_proba:
            # Exploration: Randomly choose an action
            action = np.random.choice(range(self.action_size))
            self.q_values = np.array([3, 3, 3])  # Used for the actions plot
            self.q_values[action] = 7  # Used for the actions plot
            return action
        else:
            # Exploitation: Choose action with the highest Q-value
            self.q_values = self.model.predict(current_state)[0]
            return np.argmax(self.q_values)

    def update_exploration_probability(self):
        """
        Update the exploration probability.
        """
        self.exploration_proba = max(0.05, self.exploration_proba * np.exp(-self.exploration_proba_decay))

    def store_episode(self, current_state, action, reward, next_state):
        """
        Store an episode in the memory buffer.
        """
        self.experience_memory.add_experience({
            "current_state": current_state,
            "action": action,  # an integer
            "reward": reward,
            "next_state": next_state,
        })

        

    def train(self, target_agent=None):
        """
        Train the model at the end of each episode.
        """
        batch_sample = self.experience_memory.get_batch_sample(self.train_size)
        trainXP_in = []
        trainXP_out = []
        current_state_list = np.array([batch_sample[i]["current_state"] for i in range(len(batch_sample))])
        next_state_list = np.array([batch_sample[i]["next_state"] for i in range(len(batch_sample))])
        
        model_pred_current_list = self.model.predict(current_state_list)
        model_pred_next_list = self.model.predict(next_state_list)

        if target_agent:
            target_pred_next_list = target_agent.model.predict(next_state_list)
            
            for q_current_state,target_pred_next,model_pred_next,experience in zip(model_pred_current_list,target_pred_next_list,model_pred_next_list,batch_sample):
                reward_t1 = experience["reward"]
                # Update Q-value using the target agent's Q-value for the next state
                utility = reward_t1 + self.gamma * target_pred_next[np.argmax(model_pred_next)]

                q_current_state[experience["action"]] = utility

                trainXP_in.append(experience["current_state"])
                trainXP_out.append(q_current_state)
        else:
            for q_current_state,model_pred_next,experience in zip(model_pred_current_list,model_pred_next_list,batch_sample):
                # Update Q-value using the agent's Q-value for the next state
                utility = reward_t1 + self.gamma * np.max(model_pred_next)

                q_current_state[experience["action"]] = utility

                trainXP_in.append(experience["current_state"])
                trainXP_out.append(q_current_state)

        trainXP_in = np.array(trainXP_in)
        print(trainXP_in.shape)
        trainXP_out = np.array(trainXP_out)

        # Train the model with the experiences
        history = self.model.fit(trainXP_in, trainXP_out, batch_size=self.batch_size , epochs = 50)

        return history.history

    def Model1(self):
        """
        Define the first neural network model.
        """
        self.model = models.Sequential()
        self.model.add(keras.Input(shape=(16, 20, self.nbChannel, 1)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 20, self.nbChannel)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(self.action_size, activation='linear'))

    def Model2(self):
        """
        Define the second neural network model.
        """
        self.model = models.Sequential()
        self.model.add(keras.Input(shape=(16, 20, self.nbChannel)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(self.action_size, activation='linear'))

    def Model3(self):
        """
        Define the second neural network model.
        """
        self.model = models.Sequential()
        self.model.add(keras.Input(shape=(16, 20, self.nbChannel)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(self.action_size, activation='linear'))