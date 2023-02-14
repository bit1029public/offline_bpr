import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, state_dim, encoder_dim):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, encoder_dim)

    def forward(self, state):
        enc = F.relu(self.l1(state))
        enc = F.relu(self.l2(enc))
        enc = F.relu(self.l3(enc))
        return enc


class Predictor(nn.Module):
    def __init__(self, state_dim, encoder_dim):
        super(Predictor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, encoder_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            save_eigen=False,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            opt_predictor=True,
    ):
        encoder_dim = 256
        self.encoder = Encoder(state_dim, encoder_dim).to(device)
        self.predictor = Predictor(encoder_dim, action_dim).to(device)

        self.target_encoder = copy.deepcopy(self.encoder)

        self.save_eigen = save_eigen

        if opt_predictor == 'True':
            self.encoder_optimizer = torch.optim.Adam(
                list(self.encoder.parameters())+list(self.predictor.parameters()), lr=3e-4)
        else:
            self.encoder_optimizer = torch.optim.Adam(
                list(self.encoder.parameters()), lr=3e-4)

        self.actor = Actor(encoder_dim, action_dim, max_action).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()), lr=3e-4)

        self.critic = Critic(encoder_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0
        self.it = 0

        self.effective_eigen = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = self.encoder(state)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)
        if self.total_it < 1e5:
            state_ = self.encoder(state)
            behavioral_action = self.predictor(state_)
            behavioral_action = F.normalize(behavioral_action, dim=1, p=2)
            action_ = F.normalize(action, dim=1, p=2)
            enc_loss = F.mse_loss(behavioral_action, action_)

            self.encoder_optimizer.zero_grad()
            enc_loss.backward()
            self.encoder_optimizer.step()

            if (self.total_it % 1000 == 0) and self.save_eigen:
                corr = torch.einsum('bi,bj->ij', [state_, state_])
                corr_ = corr.clone()
                eigval, eigvec = torch.eig(corr_)
                self.effective_eigen.append(torch.sum(eigval > 0.001))
            return

        state_ = self.encoder(state).detach()
        next_state_ = self.encoder(next_state).detach()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            na = self.actor_target(next_state_)
            next_action = (
                na + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_, action)

        # Compute critic loss

        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            state_ = self.encoder(state).detach()
            pi = self.actor(state_)
            Q = self.critic.Q1(state_, pi)
            lmbda = self.alpha/Q.abs().mean().detach()

            actor_loss = - lmbda * Q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        if (self.total_it % self.policy_freq == 0) and (self.total_it % 10000 == 0):
            print(actor_loss)
            print(critic_loss)

    def save(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

        torch.save(self.encoder.state_dict(), filename + "_encoder")
        torch.save(self.predictor.state_dict(), filename + "_predictor")
        torch.save(self.encoder_optimizer.state_dict(),
                   filename + "_encoder_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.encoder_optimizer.load_state_dict(
            torch.load(filename + "_encoder_optimizer"))

        self.predictor.load_state_dict(torch.load(filename + "_predictor"))

    def load_enc(self, filename):

        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.predictor.load_state_dict(torch.load(filename + "_predictor"))
