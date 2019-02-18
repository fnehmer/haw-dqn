## Solving OpenAI Gym CartPole game with a DQN (Reinforcement Learning)
For my assignment at the HAW Hamburg course "Intelligente Systeme" (engl: intellegent systems) I needed to do 3 practical researches of self-chosen topics in the areas "Searching", "Learning" and "NLP". For "Learning" I decided to assess Deep Q-Learning with the CartPole game example of OpenAI Gym.

### Cart Pole Game
The goal of the OpenAI CartPole game is to balance out a Pole on a moving cart.
[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)

In this repo there is the code for a solution using DQN. It is inspired by [Keon Kim](https://github.com/keon). I added a lot of comments in german (for the assignment) and implemented some pyPlots for visualization of the results, aswell as modifications of the training.

### DQN
The goal of a DQN is to approximate Q*.
Q* represents the max. Q Value, which is the maximum expected future reward of state-acion combinations regarding an action.
