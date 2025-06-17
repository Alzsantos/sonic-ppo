#!/usr/bin/env python
# coding: utf-8

import retro
import cv2
import gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from gym import Wrapper
import os
import gzip

SCALE_FACTOR = 2
SKIP = 1  # Número de frames a serem pulados

# Classe de pré-processamento (igual a usada no treino)
class PreprocessamentoAmbienteSonic(Wrapper):
    def __init__(self, env, dim_obs=(112, 160)):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*dim_obs, 3), dtype=np.uint8)
        self.dim_obs = dim_obs

    def redimensiona_observacao(self, obs):
        return cv2.resize(obs, (self.dim_obs[1], self.dim_obs[0]), interpolation=cv2.INTER_AREA)

    def reset(self):
        obs = self.env.reset()
        return self.redimensiona_observacao(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.redimensiona_observacao(obs)
        return obs, reward, done, info

# Função para criar o ambiente com wrapper
def criar_ambiente():
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    env = PreprocessamentoAmbienteSonic(env)
    return env

# Testa o modelo
def testa_modelo(caminho_modelo):
    env = criar_ambiente()

    


    modelo = RecurrentPPO.load(caminho_modelo)
    obs = env.reset()
    lstm_states = None
    done = False
    recompensa_total = 0

    gravou = False

    while not done:
        action, lstm_states = modelo.predict(obs, state=lstm_states, episode_start=np.array([done]), deterministic = True)

        for _ in range(SKIP):
            obs, reward, done, info = env.step(action)
            if done:
                break

        recompensa_total += reward

        frame = env.render(mode='rgb_array')



        frame_bgr = cv2.cvtColor(env.render(mode='rgb_array'), cv2.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[:2]
        resized_frame = cv2.resize(frame_bgr, (width * SCALE_FACTOR, height * SCALE_FACTOR))
        
        cv2.imshow('Sonic', resized_frame)
        cv2.waitKey(1)

        print(info['x'])
       

    print(f"Recompensa total: {recompensa_total}")
    env.close()

# Caminho do modelo treinado
CAMINHO_MODELO = './checkpoint_sonic_ppo/MelhorModelo.zip'
testa_modelo(CAMINHO_MODELO)
