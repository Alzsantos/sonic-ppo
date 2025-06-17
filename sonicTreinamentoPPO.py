#!/usr/bin/env python
# coding: utf-8

import retro
import numpy as np
import cv2
import gym
import os
import logging
from datetime import datetime

from gym import Wrapper
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from sb3_contrib import RecurrentPPO


# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHECKPOINT_DIR = "./checkpoint_sonic_ppo/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def grava_arquivo(conteudo):
    with open(os.path.join(CHECKPOINT_DIR, 'recompensa.txt'), 'a') as arquivo:
        arquivo.writelines(conteudo + '\n')


class PreprocessamentoAmbienteSonic(Wrapper):
    def __init__(self, env, dim_obs=(112, 160)):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*dim_obs, 3), dtype=np.uint8)
        self.dim_obs = dim_obs
        self.QTD_LIM_ACAO_PARADO = 10
        self.RECOMPENSA_BASE = 4
        self.inicializar_valores()

    def redimensiona_observacao(self, obs):
        return cv2.resize(obs, (self.dim_obs[1], self.dim_obs[0]), interpolation=cv2.INTER_AREA)

    def inicializar_valores(self):
        self.quantidade_argolas = 0
        self.quantidade_vidas = 3
        self.quantidade_pontos = 0
        self.zona = 1
        self.ato = 1
        self.state_inicial = None
        self.qtd_acao_parado = 0
        self.posicao_x = 80
        self.controle_x = 100
        self.velocidade = 0
        logging.info('Valores iniciais resetados')

    def atualizar_vidas(self, info):
        if info['lives'] < self.quantidade_vidas:
            self.inicializar_valores()
            self.zona = info['zone']
            self.ato = info['act']
        self.quantidade_vidas = info['lives']

    def calcular_recompensa_movimentacao(self, info):
        posicao_x = info['x']
        recompensa = posicao_x * self.quantidade_vidas
        self.velocidade = posicao_x - self.posicao_x
        if self.velocidade > 20:
            self.velocidade = 1
        self.posicao_x = posicao_x
        if posicao_x > self.controle_x:
            logging.info(f"Velocidade: {self.velocidade} | Posicao: {posicao_x}")
            self.controle_x += 100
        return recompensa

    def calcular_recompensa_argolas(self, info):
        quantidade_argolas = info["rings"]
        recompensa = 0
        if quantidade_argolas >= self.quantidade_argolas:
            recompensa = quantidade_argolas * 70 * self.quantidade_vidas
        elif quantidade_argolas < self.quantidade_argolas:
            recompensa = -1
        self.quantidade_argolas = quantidade_argolas
        return recompensa

    def calcular_recompensa_pontos(self, info):
        quantidade_pontos = info['score']
        recompensa = 0
        if quantidade_pontos >= self.quantidade_pontos:
            recompensa = quantidade_pontos * self.quantidade_vidas * 50
        self.quantidade_pontos = quantidade_pontos
        return recompensa

    def calcular_recompensa_fase_jogo(self, info):
        zona = info['zone']
        ato = info['act']
        if self.zona != zona or self.ato != ato:
            self.inicializar_valores()
        self.zona = zona
        self.ato = ato
        self.quantidade_vidas = info['lives']
        return (zona * 10) + (ato + 1)

    def calcular_recompensa(self, info):
        if self.state_inicial is None:
            self.state_inicial = info
        else:
            if self.state_inicial['x'] == info['x']:
                self.qtd_acao_parado += 1
                if self.qtd_acao_parado > self.QTD_LIM_ACAO_PARADO:
                    self.qtd_acao_parado = 0
                    return -10
            else:
                self.qtd_acao_parado = 0
            self.state_inicial = info

        self.atualizar_vidas(info)
        recompensa_posicao = self.calcular_recompensa_movimentacao(info)
        recompensa_argolas = self.calcular_recompensa_argolas(info)
        recompensa_pontos = self.calcular_recompensa_pontos(info)
        recompensa_fase_jogo = self.calcular_recompensa_fase_jogo(info)

        if recompensa_argolas != 0:
            grava_arquivo("Recompensa Argolas: " + str(recompensa_argolas))
            logging.info(f"Recompensa Argolas: {recompensa_argolas}")

        if recompensa_pontos != 0:
            grava_arquivo("Recompensa Pontos:" + str(recompensa_pontos))

        recompensa_total = ((recompensa_posicao + recompensa_pontos + recompensa_argolas) * recompensa_fase_jogo) * (self.velocidade / 100)

        if recompensa_total > 1000:
            logging.info(f"Recompensa Total: {recompensa_total}")

        return recompensa_total

    def reset(self):
        obs = self.env.reset()
        obs = self.redimensiona_observacao(obs)
        self.inicializar_valores()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.calcular_recompensa(info)
        obs = self.redimensiona_observacao(obs)
        if done:
            logging.info("***** TERMINOU *****")
        return obs, reward, done, info


class SaveOnBestRewardCallback(BaseCallback):
    def __init__(self, env, check_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_reward = -np.inf
        self.env = env

    def _get_nome_arquivo(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"model_{timestamp}"

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                last_reward = self.model.ep_info_buffer[-1]['r']
                if last_reward > self.best_reward:
                    self.best_reward = last_reward
                    self.model.save(os.path.join(self.save_path, self._get_nome_arquivo()))
                    if self.verbose:
                        print(f"Novo melhor modelo salvo com recompensa: {last_reward:.2f}")
        return True


def criar_ambiente_treinamento():
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    env = PreprocessamentoAmbienteSonic(env)
    return env


def main():
    env = criar_ambiente_treinamento()

    best_reward_callback = SaveOnBestRewardCallback(
        env=env,
        check_freq=1000,
        save_path=CHECKPOINT_DIR,
        verbose=2
    )

    checkpoint_callback = CheckpointCallback(save_freq=15000, save_path=CHECKPOINT_DIR, name_prefix='sonic_ppo')

    lr_schedule = get_linear_fn(2.5e-4, 1e-5, 1.0)

    modelo = RecurrentPPO.load('./checkpoint_sonic_ppo/sonic_ppo_1725000_steps_Melhor.zip')
    modelo.batch_size = 256
    modelo.ent_coef = 0.1
    modelo.max_grad_norm = 0.5
    modelo.vf_coef = 0.5
    modelo.policy_kwargs=dict(
            lstm_hidden_size=256,
            share_features_extractor=True
        )
    modelo.set_env(env)

    modelo.learn(
        total_timesteps=25_000_000,
        callback=CallbackList([checkpoint_callback, best_reward_callback]),
        reset_num_timesteps=False
    )

    modelo.save('sonic_ppo_final')


if __name__ == "__main__":
    main()





