
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Predictor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Predictor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class NoShiftAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class VectorQuantizerEMA(nn.Module):
    def __init__(self, embedding_dim=128, num_embeddings=120, groups=1, commitment_cost=0.25, decay=0.95, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__() 
       
        print('num codes', num_embeddings)

        self._embedding_dim = embedding_dim // groups
        self.groups = groups
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):

        x, y = inputs.size()
        inputs = inputs.reshape(x, 1, y)
        # BCHW -> BHWC
        #inputs = inputs.permute(0, 2, 3, 1).contiguous()
        bs, t, m = inputs.shape
        inputs = inputs.reshape(bs, t, self.groups, m // self.groups)
        inputs = inputs.permute(0, 2, 1, 3)
        inputs = inputs.reshape(bs * self.groups, t, m // self.groups)
        input_shape = inputs.shape
        

        # Flatten 
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # May be we can even use cosine distance.
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        distances *= self._ema_cluster_size.unsqueeze(0).repeat(distances.shape[0], 1)
        #distances *= torch.sqrt(torch.sqrt(self._ema_cluster_size.unsqueeze(0).repeat(distances.shape[0], 1)))
        #distances *= torch.exp((self._ema_cluster_size.unsqueeze(0).repeat(distances.shape[0], 1) / 20.0))


        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        #  EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
 
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
 
       
        # BHWC -> BCHW
        quantized = quantized.reshape(bs, self.groups, t, m // self.groups)
        quantized = quantized.permute(0, 2, 1, 3)
        quantized = quantized.reshape(bs, t, m)

        encoding_indices = encoding_indices.reshape(bs, self.groups, t)

        z_embed = quantized.contiguous()
        x, v, y = z_embed.shape
        z_embed = z_embed.reshape(x, y)

        x, y, v = encoding_indices.size()
        encoding_indices = encoding_indices.reshape(x, y)

        return z_embed, loss, encoding_indices
    
class DrQAcstateAgent:
    def __init__(self, obs_shape, action_shape, max_action, num_protos, groups, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 offline=False, bc_weight=2.5, augmentation=RandomShiftsAug(pad=4),
                 use_bc=True):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.offline = offline
        self.bc_weight = bc_weight
        self.use_bc = use_bc
        self.num_protos = num_protos
        self.groups = groups
        # TODO: need to be specified
        self.kl_penalty = 0.01

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim//2, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim//2, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim//2, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.post_enc = nn.Linear(self.encoder.repr_dim//2, self.encoder.repr_dim//2).to(self.device)
        self.vq_layer = VectorQuantizerEMA(self.encoder.repr_dim//2, num_embeddings=self.num_protos, groups=self.groups).to(self.device)
        
        self.k_embedding = nn.Embedding(20, self.encoder.repr_dim//2).to(self.device)
        self.action_mlp = nn.Sequential(nn.Linear(3 * (self.encoder.repr_dim//2), \
                            2 * self.encoder.repr_dim//2), \
                            nn.BatchNorm1d(2*self.encoder.repr_dim//2), \
                            nn.GELU(), \
                            nn.Linear(2 * self.encoder.repr_dim//2, self.encoder.repr_dim//2)).to(self.device)
        self.action_mlp_1 = nn.Linear(self.encoder.repr_dim//2, action_shape[0]).to(self.device)
        
        # optimizers
        self.encoder_opt = torch.optim.Adam(list(self.encoder.parameters())+\
                                list(self.post_enc.parameters())+\
                                list(self.vq_layer.parameters())+\
                                list(self.k_embedding.parameters())
                                +\
                                list(self.action_mlp.parameters())+\
                                list(self.action_mlp_1.parameters())
                                , lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        
        # data augmentation
        self.aug = augmentation

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        
        obs = self.encoder(obs.unsqueeze(0))
        mu_obs = obs[:,:self.encoder.repr_dim//2]
        obs = self.post_enc(mu_obs)
        obs, _, _ = self.vq_layer(obs)
        
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]


    def update_encoder(self, obs, obs_k, action, k):
        
        # width = obs.shape[3]
        # # # TODO: this is copied from ac-state
        # iL = obs[:,:,:,:width//2]
        # iR = obs[:,:,:,width//2:]
        # kL = obs_k[:,:,:,:width//2]
        # kR = obs_k[:,:,:,width//2:]

        # # iC = torch.randint(0,2,size=(obs.shape[0],1,1,1)).float().cuda().repeat(1,obs.shape[1],obs.shape[2],width//2)
        # # kC = torch.randint(0,2,size=(obs.shape[0],1,1,1)).float().cuda().repeat(1,obs.shape[1],obs.shape[2],width//2)

        # obs = iL + iR
        # obs_k = kL + kR
        # # # TODO: dim=0 in ac-state, I change this to dim=-1, need to be verified
        
        obs = torch.cat((obs, obs_k), dim = 0)
        
        obs = self.encoder(obs)
        
        mu_s = obs[:,:self.encoder.repr_dim//2]
        std_s = torch.nn.functional.softplus(obs[:,self.encoder.repr_dim//2:])
        kl_loss = (mu_s**2 + std_s**2 - 2*torch.log(std_s)).sum(dim=1).mean() * self.kl_penalty
        obs = mu_s + torch.randn_like(std_s) * std_s
        obs = self.post_enc(obs)
        z_vq, vq_loss, _ = self.vq_layer(obs)
        obs, obs_k = torch.chunk(obs, 2, dim = 0)
        
        k = self.k_embedding(k)
        obs_cat = torch.cat((obs, obs_k, k), dim = 1)
        obs_cat = self.action_mlp(obs_cat)
        action_pred = self.action_mlp_1(obs_cat)
        
        inverse_loss = F.mse_loss(action_pred, action)
        
        encoder_loss = vq_loss + kl_loss + inverse_loss
        self.encoder_opt.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_opt.step()
        metrics = dict()
        return metrics
        
        
        
        
        


    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward.float() + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, behavioural_action=None):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_policy_improvement_loss = -Q.mean()

        actor_loss = actor_policy_improvement_loss

        # offline BC Loss
        if self.offline:
            actor_bc_loss = F.mse_loss(action, behavioural_action)
            # Eq. 5 of arXiv:2106.06860
            lam = self.bc_weight / Q.detach().abs().mean()
            if self.use_bc:
                actor_loss = actor_policy_improvement_loss * lam + actor_bc_loss
            else:
                actor_loss = actor_policy_improvement_loss * lam

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_policy_improvement_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            if self.offline:
                metrics['actor_bc_loss'] = actor_bc_loss.item()

        return metrics

    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        obs, action, reward, discount, next_obs, k_step, obs_k = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        obs_k = self.aug(obs_k.float())
        
        metrics.update(self.update_encoder(obs, obs_k, action, k_step))
        
        
        
        
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
            
        mu_obs = obs[:,:self.encoder.repr_dim//2]
        next_mu_obs = next_obs[:,:self.encoder.repr_dim//2]
        obs = self.post_enc(mu_obs)
        next_obs = self.post_enc(next_mu_obs)
        obs, _, _ = self.vq_layer(obs)
        next_obs, _, _ = self.vq_layer(next_obs)
        
            

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        if self.offline:
            metrics.update(self.update_actor(obs.detach(), step, action.detach()))
        else:
            metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
