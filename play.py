import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
from time import time
from statistics import mean
import boto3
import json

import os
import gym
import numpy as np

class BuzzAldrin(object):
    def __init__(self, id_):
        self.env = gym.make("LunarLander-v2")
        self.id = id_
        self.filepath = f"models/{self.id}/"
        self.play_log = []

        self.s3_backup = False

    def choose_greedy_action(self, actions):
        return np.argmax(actions)

    def play(self, filename, n, render=False):
        print(f"Starting {filename}")
        model = load_model(self.filepath + filename)

        for T in range(n):
            s = self.env.reset()
            done = False

            steps = 0
            cumulative_reward = 0

            while not done:
                if render:
                    self.env.render()

                steps += 1

                action = np.argmax(model.predict(s.reshape(1, 8))[0])
                new_s, reward, done, _ = self.env.step(action)

                cumulative_reward += reward
                s = new_s

            self.play_log.append([T, steps, cumulative_reward])
        
        print(f"Average Reward over {n} Simulations: {sum([x[2] for x in self.play_log]) / n}")

    def save_rewards(self, filename):
        key = self.filepath + filename
        with open(key, 'w', encoding='utf-8') as fh:
            json.dump(self.play_log, fh, indent=4)

        if self.s3_backup:
            client = boto3.client('s3')
            resp = client.put_object(Body=json.dumps(self.play_log, sort_keys=True, indent=4), Bucket="mscully8-gatech", Key=key)

if __name__ == "__main__":
    #My Best Model
    uuid = '82ba7a18-d87e-11eb-a531-02362b9e3ec8'
    filename = f"model_850.hd5"
    SIMULATIONS = 1

    agent = BuzzAldrin(id_=uuid)
    agent.play(filename, SIMULATIONS, render=True)

