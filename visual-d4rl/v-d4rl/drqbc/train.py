
from utils import load_offline_dataset_into_buffer
from video import TrainVideoRecorder, VideoRecorder
from numpy_replay_buffer import EfficientReplayBuffer
from logger import Logger
import utils
import dmc
from dm_env import specs
import torch
import numpy as np
import hydra
from pathlib import Path
import os
import warnings
import time
warnings.filterwarnings('ignore', category=DeprecationWarning)


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'


torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.max_action = int(action_spec.maximum)
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        print('=======================')
        print(self.cfg)
        print('=======================')
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._pretrain_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, offline=self.cfg.offline,
                             distracting_eval=self.cfg.eval_on_distracting, multitask_eval=self.cfg.eval_on_multitask)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, self.cfg.distracting_mode)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, self.cfg.distracting_mode)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_buffer = EfficientReplayBuffer(self.cfg.replay_buffer_size,
                                                   self.cfg.batch_size,
                                                   self.cfg.nstep,
                                                   self.cfg.discount,
                                                   self.cfg.frame_stack,
                                                   data_specs)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

        self.eval_on_distracting = self.cfg.eval_on_distracting
        self.eval_on_multitask = self.cfg.eval_on_multitask

    @property
    def global_step(self):
        return self._global_step

    @property
    def pretrain_step(self):
        return self._pretrain_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def pretrain_frame(self):
        return self.pretrain_step * self.cfg.action_repeat

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def eval_pretrain(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.pretrain_act(time_step.observation,
                                                     self.global_step,
                                                     eval_mode=True)
                time_step = self.eval_env.step(action)
                total_reward += time_step.reward
                step += 1

            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def eval_distracting(self, record_video):
        distraction_modes = ['easy', 'medium', 'hard',
                             'fixed_easy', 'fixed_medium', 'fixed_hard']
        if not hasattr(self, 'distracting_envs'):
            self.distracting_envs = []
            for distraction_mode in distraction_modes:
                env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                               self.cfg.action_repeat, self.cfg.seed, distracting_mode=distraction_mode)
                self.distracting_envs.append(env)
        for env, env_name in zip(self.distracting_envs, distraction_modes):
            self.eval_single_env(env, env_name, record_video)

    def eval_multitask(self, record_video):
        multitask_modes = [f'len_{i}' for i in range(1, 11, 1)]
        if not hasattr(self, 'multitask_envs'):
            self.multitask_envs = []
            for multitask_mode in multitask_modes:
                env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                               self.cfg.action_repeat, self.cfg.seed, multitask_mode=multitask_mode)
                self.multitask_envs.append(env)
        for env, env_name in zip(self.multitask_envs, multitask_modes):
            self.eval_single_env(env, env_name, record_video)

    def eval_single_env(self, env, env_name, save_video):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = env.reset()
            self.video_recorder.init(
                env, enabled=((episode == 0) and save_video))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = env.step(action)
                self.video_recorder.record(env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{env_name}_{self.global_frame}.mp4')

        self.logger.log(f'eval/{env_name}_episode_reward',
                        total_reward / episode, self.global_frame)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)

        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        # only in distracting evaluation mode
        eval_save_vid_every_step = utils.Every(self.cfg.eval_save_vid_every_step,
                                               self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.eval_on_distracting:
                    self.eval_distracting(
                        eval_save_vid_every_step(self.global_step))
                if self.eval_on_multitask:
                    self.eval_multitask(
                        eval_save_vid_every_step(self.global_step))
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)
            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def train_offline(self, offline_dir):
        # Open dataset, load as memory buffer
        load_offline_dataset_into_buffer(Path(offline_dir), self.replay_buffer, self.cfg.frame_stack,
                                         self.cfg.replay_buffer_size)

        if self.replay_buffer.index == -1:
            raise ValueError('No offline data loaded, check directory.')

        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames, 1)
        pretrain_until_step = utils.Until(self.cfg.pretrain_num_frames, 1)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, 1)
        show_train_stats_every_step = utils.Every(
            self.cfg.show_train_stats_every_frames, 1)
        # only in distracting evaluation mode
        eval_save_vid_every_step = utils.Every(self.cfg.eval_save_vid_every_step,
                                               self.cfg.action_repeat)

        metrics = None
        step = 0
        if 'BPR' in self.cfg.agent['_target_']:
            while pretrain_until_step(self.pretrain_step):
                previous_time = time.time()
                if show_train_stats_every_step(self.pretrain_step):
                    # wait until all the metrics schema is populated
                    if self.pretrain_step % 10000 == 0:
                        tn = time.time()
                        print('pretraining step finished:', str(self.pretrain_step), ' steps',
                              ', time cost:', tn-previous_time)
                        previous_time = tn
                    # if eval_every_step(self._pretrain_step):
                    #     self.logger.log('eval_total_time', self.timer.total_time(),
                    #                     self.global_frame)
                    #     self.eval_pretrain()
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                step += 1
                self.agent.pretrain(self.replay_buffer, self.pretrain_step)
                # if show_train_stats_every_step(self.pretrain_step):
                #     self.logger.log_metrics(metrics, self.pretrain_frame, ty='pretrain')
                self._pretrain_step += 1

        while train_until_step(self.global_step):
            if show_train_stats_every_step(self.global_step):
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', step / elapsed_time)
                        log('total_time', total_time)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)
                    step = 0
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
            step += 1
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.eval_on_distracting:
                    self.eval_distracting(
                        eval_save_vid_every_step(self.global_step))
                if self.eval_on_multitask:
                    self.eval_multitask(
                        eval_save_vid_every_step(self.global_step))
                self.eval()

            # try to update the agent

            metrics = self.agent.update(self.replay_buffer, self.global_step)
            if show_train_stats_every_step(self.global_step):
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    if cfg.offline:
        workspace.train_offline(cfg.offline_dir)
    else:
        workspace.train()


if __name__ == '__main__':
    main()
