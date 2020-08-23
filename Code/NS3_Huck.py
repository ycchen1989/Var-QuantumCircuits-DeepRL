"""
Refer to ns3-gym for ns3 simulation.
Gawłowicz, Piotr, and Anatolij Zubow. "Ns-3 meets openai gym: The playground for machine learning in networking research." 
Proceedings of the 22nd International ACM Conference on Modeling, Analysis and Simulation of Wireless and Mobile Systems. 2019.
"""
import numpy as np

class Radio():
	def __init__(self, channel = 4, max_env_steps = 100):
		"""

		Parameters
		----------
		action :

		Returns
		-------
		ob, reward, episode_over, info : tuple
			ob (object) :
				an environment-specific object representing your observation of
				the environment.
			reward (float) :
				amount of reward achieved by the previous action. The scale
				varies between environments, but the goal is always to increase
				your total reward.
			episode_over (bool) :
				whether it's time to reset the environment again. Most (but not
				all) tasks are divided up into well-defined episodes, and done
				being True indicates the episode has terminated. (For example,
				perhaps the pole tipped too far, or you lost your last life.)
			info (dict) :
				 diagnostic information useful for debugging. It can sometimes
				 be useful for learning (for example, it might contain the raw
				 probabilities behind the environment's last state change).
				 However, official evaluations of your agent are not allowed to
				 use this for learning.
		"""


		self.donecnt = 0
		self.status = np.zeros(channel, dtype=np.int).tolist()
		print("Initial the Radio Env with",channel,  "channels -> Obs in clean channels:", self.status)
		self.t = 0
		self.channel = channel
		self.max_env_steps = max_env_steps

		
	def step(self, action):
		use_ch = self.t % self.channel
		tmp_s = np.zeros(self.channel, dtype=np.int).tolist()
		tmp_s[use_ch] = 1
		self.status = tmp_s
		self.t += 1
		self.take_action(action)
		reward = self.get_reward(action)
		ob = self.get_state()

		if (self.donecnt == 3 or self.t >= self.max_env_steps):
			episode_over = True
		else:
			episode_over = False
		
		return ob, reward, episode_over

#     def reset(self):
#         pass

	def get_state(self):
		return self.status

	def take_action(self, action):
		if self.status[action] == 1:
			self.donecnt+=1
			
	def get_reward(self, action):
		""" Reward is given for. """
		if self.status[action] == 1:
			return -1
		else:
			return 1
