import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from time import time
from statistics import mean
import uuid
import boto3
import json
import pandas as pd

import os
import gym
import numpy as np

class LunarLanderLearner(object):
    def __init__(self, T, learning_rate, gamma, max_epsilon, min_epsilon, decay, tau, batch_size, queue_size):
        self.env = gym.make("LunarLander-v2")
        self.id = uuid.uuid1()

        self.T = T
        self.GAMMA = learning_rate
        self.MAX_EPSILON = max_epsilon
        self.MIN_EPSILON = min_epsilon
        self.DECAY = decay
        self.LEARNING_RATE = learning_rate
        self.TAU = tau

        self.BATCH_SIZE = batch_size
        self.QUEUE_SIZE = queue_size

        self.replay_buffer = deque(maxlen=self.QUEUE_SIZE)

        self.train_model = self.generate_model()
        self.target_model = self.generate_model()
        
        self.s3_backup = False
        self.train_log = []
        self.model_path = f"models/{self.id}/"

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def generate_model(self):
        model = Sequential([
            Dense(64, input_dim=self.env.observation_space.shape[0], activation='relu'),
            Dense(64, activation="relu"),
            Dense(self.env.action_space.n)
        ])

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE))
        return model

    def derive_epsilon(self, episode):
        return self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * (self.DECAY ** episode)

    def choose_epsilon_greedy_action(self, state, epsilon):
        if np.random.random() > epsilon:
            return np.argmax(self.train_model.predict(state.reshape(1, 8))[0])
        else:
            return self.env.action_space.sample()

    def learn(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        #Select 128 samples at random ([s, a, r, s_prime, done])
        batch = np.array(random.sample(self.replay_buffer, self.BATCH_SIZE))

        #Generate predictions for s' from both moedls
        train_predictions = self.train_model.predict(np.stack(batch[:, 3], axis=0))
        target_predictions = self.target_model.predict(np.stack(batch[:, 3], axis=0))
        
        states = np.stack(batch[:, 0], axis=0)
        actions = np.array(batch[:, 1], dtype=int)
        rewards = batch[:, 2]

        #Convert the true/false done column into 1s and 0s
        done = np.logical_not(np.copy(batch[:, 4])).astype(int)

        #Determine the best action from the training predictions
        train_best_actions = np.argmax(train_predictions, axis=1)

        #Get the expected value of those best actions from the target predictions
        s_prime_q_vals = target_predictions[np.arange(self.BATCH_SIZE), train_best_actions]

        #Generate predictions of s using the training model
        s_q_vals = self.train_model.predict(states)

        #Update the predictions for samples for actions that actually were taken to be the reward plus the discounted expected value of s'
        s_q_vals[np.arange(self.BATCH_SIZE), actions] = rewards + (self.GAMMA * s_prime_q_vals * done)

        #Feed the predictions to the NN
        self.train_model.fit(states, s_q_vals, epochs=1, verbose=0)

    def move_train_to_target(self):
        if self.TAU == 1:
            self.target_model.set_weights(self.train_model.get_weights())
        else:
            training_weights = self.train_model.get_weights()
            target_weights = self.target_model.get_weights()

            for i in range(len(training_weights)):
                target_weights[i] = (training_weights[i] * self.TAU) + (target_weights[i] * (1 - self.TAU))

            self.target_model.set_weights(target_weights)

    def save_hyperparamters(self):
        filename = "hyperparameters.json"

        hyperparameters = {
            "T": self.T,
            "GAMMA": self.GAMMA,
            "MAX_EPSILON": self.MAX_EPSILON,
            "MIN_EPSILON": self.MIN_EPSILON,
            "DECAY": self.DECAY,
            "LEARNING_RATE": self.LEARNING_RATE,
            "TAU": self.TAU,
            "BATCH_SIZE":  self.BATCH_SIZE,
            "QUEUE_SIZE": self.QUEUE_SIZE,
        }
        
        with open(self.model_path + filename, "w") as fh:
            json.dump(hyperparameters, fh)

        if self.s3_backup:
            client = boto3.client('s3')
            client.put_object(Body=json.dumps(hyperparameters, sort_keys=True, indent=4), Bucket="mscully8-gatech", Key=self.model_path + filename)
            print("Hyperparameters Saved!")

    def save_model(self, episode):
        filename = f"model_{episode}.hd5"

        self.train_model.save(self.model_path + filename)
        print("Model Saved!")

        if self.s3_backup:
            client = boto3.client('s3')
            client.upload_file(Filename=self.model_path+filename, Bucket="mscully8-gatech", Key=self.model_path+filename)
            print("Model Saved to S3!")

    def save_train_log(self):
        filename = "training_log.json"
        df = pd.DataFrame(self.train_log, columns=["T", "Epsilon", "Steps", "Total Reward", "Duration"])
        json = df.to_json(indent=4)

        with open(self.model_path + filename, "w") as fh:
            fh.write(json)
        
        if self.s3_backup:                
            client = boto3.client('s3')
            client.upload_file(Filename=self.model_path+filename, Bucket="mscully8-gatech", Key=self.model_path+filename)

    def write_to_text_file(self, msg):
        with open(f"{os.getcwd()}/{self.id}.txt", 'a') as fh:
            fh.write(msg)
            fh.write('\n')

    def solve(self):
        self.save_hyperparamters()
        print(f"Begin Solving {self.id}")
        print(f"Hyperparameters -> Gamma: {self.GAMMA}, Decay: {self.DECAY}, Learning Rate: {self.LEARNING_RATE}, Batch Size: {self.BATCH_SIZE}")
        
        reward_array = deque(maxlen=100)
        max_reward = 0
        for T in range(self.T):
            start_time = time()
            epsilon = self.derive_epsilon(T)

            s = self.env.reset()
            done = False

            steps = 0
            cumulative_reward = 0

            while not done:
                steps += 1

                action = self.choose_epsilon_greedy_action(s, epsilon)
                new_s, reward, done, _ = self.env.step(action)

                self.replay_buffer.append([s, action, reward, new_s, done])
                self.learn()

                s = new_s

                cumulative_reward += reward

            self.move_train_to_target()
            reward_array.append(cumulative_reward)
            self.train_log.append([T, epsilon, steps, cumulative_reward, time() - start_time]) 
            self.write_to_text_file(f"Iteration: {T + 1}, Epsilon: {round(epsilon, 3)}, Steps: {steps}, Current Reward: {round(cumulative_reward, 3)}, Average Reward: {round(mean(reward_array), 3)}, Duration: {round(time() - start_time, 3)}")

            if cumulative_reward > 0 and cumulative_reward < (0.8 * max_reward):
                print("Model has peaked!")
                break

            if T >= self.QUEUE_SIZE and cumulative_reward > max_reward:
                print("New Max Reward!")
                max_reward = cumulative_reward

            if T > 0 and T % 50 == 0:
                self.save_model(T)
                self.save_train_log()

if __name__ == "__main__":
    T = 1000
    GAMMA = 0.99
    MAX_EPSILON = 1
    MIN_EPSILON = 0.01
    DECAY = 0.993
    LEARNING_RATE = 0.0001
    TAU = 1

    BATCH_SIZE = 128
    QUEUE_SIZE = 20000
    
    agent = LunarLanderLearner(T=T, learning_rate=LEARNING_RATE, gamma=GAMMA, min_epsilon=MIN_EPSILON, max_epsilon=MAX_EPSILON, decay=DECAY, tau=TAU, batch_size=BATCH_SIZE, queue_size=QUEUE_SIZE)
    agent.solve()
