import numpy as np
import tensorflow as tf

from mbpo.models.fc import FC
from mbpo.models.bnn import BNN
from mbpo.models.causal_bnn import CausalBNN
from mbpo.models.single_dim_fc import SingleDimFC

def construct_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):
	print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
	params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
	# num_networks是model ensemble的数量
	model = BNN(params)

	# 第一层必须指定input，后面层的input可以自动计算
	model.add(FC(hidden_dim, input_dim=obs_dim+act_dim, activation="swish", weight_decay=0.000025))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
	model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
	model.add(FC(obs_dim+rew_dim, weight_decay=0.0001))
	model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
	return model


def construct_casual_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):
	print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
	params = {'name': 'CausalBNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
	# num_networks是model ensemble的数量
	casual_model = CausalBNN(params, act_dim, obs_dim, rew_dim)

	# 第一层必须指定input，后面层的input可以自动计算
	casual_model.add(SingleDimFC(hidden_dim, input_dim=obs_dim + act_dim, activation="swish", weight_decay=0.000025))
	casual_model.add(SingleDimFC(hidden_dim, activation="swish", weight_decay=0.00005))
	casual_model.add(SingleDimFC(hidden_dim, activation="swish", weight_decay=0.000075))
	casual_model.add(SingleDimFC(1, weight_decay=0.0001))
	casual_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
	return casual_model


def format_samples_for_training(samples):
	obs = samples['observations']
	act = samples['actions']
	next_obs = samples['next_observations']
	rew = samples['rewards']
	# 预测obs之间的残差
	delta_obs = next_obs - obs
	inputs = np.concatenate((obs, act), axis=-1)
	outputs = np.concatenate((rew, delta_obs), axis=-1)
	return inputs, outputs


def reset_model(model):
	model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
	model.sess.run(tf.initialize_vars(model_vars))


if __name__ == '__main__':
	model = construct_model()
