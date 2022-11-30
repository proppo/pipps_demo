from gym.envs.registration import register

register(id='pipps-cartpole-swingup-v0',
         entry_point='pipps.envs.cartpole:CartPoleSwingUpEnv',
         kwargs={'balance': False})

register(id='pipps-cartpole-balance-v0',
         entry_point='pipps.envs.cartpole:CartPoleSwingUpEnv',
         kwargs={'balance': True})

register(id='pipps-cartpole-swingup-tip-v0',
         entry_point='pipps.envs.cartpole:CartPoleSwingUpEnv',
         kwargs={
             'balance': False,
             'tip_cost': True
         })

register(id='pipps-cartpole-balance-tip-v0',
         entry_point='pipps.envs.cartpole:CartPoleSwingUpEnv',
         kwargs={
             'balance': True,
             'tip_cost': True
         })

register(id='pipps-dm-cartpole-swingup-v0',
         entry_point='pipps.envs.dm_control.cartpole:DmCartPoleSwingUpEnv',
         kwargs={'balance': False})

register(id='pipps-dm-cartpole-balance-v0',
         entry_point='pipps.envs.dm_control.cartpole:DmCartPoleSwingUpEnv',
         kwargs={'balance': True})
