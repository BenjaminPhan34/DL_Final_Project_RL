import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
import time
import numpy as np


class VisualizationBoard:
    def __init__(self,filename,disable_visualization):
        self.disable_visualization = disable_visualization
        plt.ion()
        plt.style.use('dark_background')
        self.window_size = 10
        self.filename = filename
        self.figDisplay, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))

        self.ax1.set_xlabel('Episode', color='lightgray', fontsize=14)
        self.ax1.set_ylabel('Reward', color='lightgray', fontsize=14)
        self.ax1.grid(axis='y', color='gray', linewidth=0.5)
        self.ax1.tick_params(axis='both', which='both', labelsize=12, colors='lightgray')
        self.ax1.set_title("Reward Plot with Moving Average", color='lightgray', fontsize=16)
        moving_average_label = f'Moving Average ({self.window_size} episodes)'
        custom_legend_entry = plt.Line2D([0], [0], color='orange', markersize=10, label=moving_average_label)
        self.ax1.legend(handles=[custom_legend_entry])

        self.ax2.set_xlabel('Actions', color='lightgray', fontsize=14)
        self.ax2.set_ylabel('b_actions', color='lightgray', fontsize=14)
        self.ax2.set_title('What is the agent action', color='lightgray', fontsize=16)
        self.ax2.tick_params(axis='both', which='both', labelsize=12, colors='lightgray')
        self.ax2.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
        legend_entries = [
            mpatches.Patch(color='darkgrey', label='Eliminated decision'),
            mpatches.Patch(color='lightgreen', label='Agent decision'),
            mpatches.Patch(color='indianred', label='Random decision')
        ]
        self.ax2.legend(handles=legend_entries, loc='lower left')
        
        if not self.disable_visualization:
            self.figDisplay.canvas.draw()
        self.actionBox = self.figDisplay.canvas.copy_from_bbox(self.ax2.bbox)
        self.RewardBox = self.figDisplay.canvas.copy_from_bbox(self.ax1.bbox)
        self.nb_episodes = 0
        self.save_rewards_plots()

    def full_extent(self,ax, pad=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""
        # For text objects, we need to draw the figure first, otherwise the extents
        # are undefined.
        items = ax.get_xticklabels() + ax.get_yticklabels() 
        #items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        items += [ax, ax.title]
        bbox = Bbox.union([item.get_window_extent() for item in items])
        return bbox.expanded(1.0 - 0.06, 1.0 + 0.2)
    
    def save_rewards_plots(self, title="Reward Plot", save_path=None):
        self.figDisplay.canvas.restore_region(self.RewardBox)
        with open(self.filename, 'r') as file:
            data = [json.loads(line) for line in file]

        reward_data = [entry["reward_episode"] for entry in data]
        episodes = range(1, len(reward_data) + 1)
        self.nb_episodes = len(reward_data)

        if save_path:
            for line in self.ax1.get_lines():
                line.remove()


        # Plot the rewards
        (ln,) = self.ax1.plot(episodes, reward_data, color='lightblue', markersize=8, markeredgecolor='white', linewidth=2,alpha=0.25)

        # Plot the moving average on top of the rewards
        if len(reward_data) >= self.window_size:
            moving_average = np.convolve(reward_data, np.ones(self.window_size) / self.window_size, mode='valid')
            ma_episodes = range(self.window_size, len(reward_data) + 1)
            (ln,) = self.ax1.plot(ma_episodes, moving_average, color='orange', linewidth=2, label=f'Moving Average ({self.window_size} episodes)')

        

        if not self.disable_visualization:
            self.ax1.draw_artist(ln)
            self.figDisplay.canvas.blit(self.ax1.bbox)
            self.figDisplay.canvas.flush_events()
        if save_path:
            if not self.disable_visualization:
                extent = self.full_extent(self.ax1).transformed(self.figDisplay.dpi_scale_trans.inverted())
                self.figDisplay.savefig(save_path, bbox_inches=extent)
            else:
                self.figDisplay.savefig(save_path)
        
        
    
    def show_action_bar(self,b_actions):
        self.figDisplay.canvas.restore_region(self.actionBox)
        # List of possible actions
        actions = ["LEFT", "RIGHT", "JUMP"]
        # Find the index of the action with the highest b_actions
        highest_b_actions_index = np.argmax(b_actions)

        # Create a list of colors, with the highest b_actions action in a different color
        colors = ['darkgrey'] * len(actions)
        colors[highest_b_actions_index] = 'lightgreen'  # Highlight the highest b_actions action
        if b_actions[highest_b_actions_index] == 7:
            colors[highest_b_actions_index] = 'indianred'
        
        normalized_actions = (b_actions - np.min(b_actions)) / (np.max(b_actions) - np.min(b_actions))

        # Clear the previous bar plot
        for rect in self.ax2.patches:
            rect.set_height(0)
        for rect in self.ax2.patches:
            rect.remove()
            
        self.ax2.set_ylim([0, normalized_actions[highest_b_actions_index] + 0.2])
        # Create a bar plot
        self.ax2.bar(actions, normalized_actions, color=colors, edgecolor='white', linewidth=2)

        # Add labels and title


        # Show the plot
        # Update the bar plot
        if not self.disable_visualization:
            self.figDisplay.canvas.blit(self.ax2.bbox)
            self.figDisplay.canvas.flush_events()
    
    def save_loss_and_accuracy_plots(self,save_path):
        with open(self.filename, 'r') as file:
            data = [json.loads(line) for line in file]

        loss_values = [entry["loss"][-1] for entry in data]
        accuracy_values = [entry["accuracy"][-1] for entry in data]

        episodes = range(1, len(data) + 1)

        fig = plt.figure(figsize=(12, 6))

        # Plot the loss
        plt.subplot(1, 2, 1)
        plt.plot(episodes, loss_values, label="Loss", color="orange")
        
        plt.title("Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True)

        # Plot the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(episodes, accuracy_values, label="Accuracy", color="cyan")
        
        plt.title("Accuracy")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.grid(True)

        if len(loss_values) >= self.window_size:
            moving_average_loss = np.convolve(loss_values, np.ones(self.window_size) / self.window_size, mode='valid')
            moving_average_accuracy = np.convolve(accuracy_values, np.ones(self.window_size) / self.window_size, mode='valid')
            plt.subplot(1, 2, 1)
            plt.plot(episodes[self.window_size - 1:], moving_average_loss, label=f"Moving Average (Loss, window={self.window_size})", color="red")
            plt.subplot(1, 2, 2)
            plt.plot(episodes[self.window_size - 1:], moving_average_accuracy, label=f"Moving Average (Accuracy, window={self.window_size})", color="blue")

        plt.tight_layout()

        # Save the plots to an image file
        plt.savefig(save_path)
        plt.close(fig)
    

    def last_loss_and_accuracy_plots(self,save_path):
        with open(self.filename, 'r') as file:
            data = [json.loads(line) for line in file]

        loss_values = data[-1]["loss"]
        accuracy_values = data[-1]["accuracy"]

        epochs = range(1, len(loss_values) + 1)

        fig = plt.figure(figsize=(12, 6))

        # Plot the loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss_values, label="Loss", color="orange")
        
        plt.title("Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True)

        # Plot the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy_values, label="Accuracy", color="cyan")
        
        plt.title("Accuracy")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.grid(True)

        plt.tight_layout()

        # Save the plots to an image file
        plt.savefig(save_path)
        plt.close(fig)