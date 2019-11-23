# bipedal-walker-adaptable-morphology

We investigate whether allowing an agent to modify its own morphology has any beneficial effect on the task finding optimal locomotive policies, on terrains of varying difficulty. Work is inspired by David Ha's and UberLabs' previous work using this agent and this environment respectively.

We use the augmented random search algorithm to optimize the policy of a bipedal walker (parametrized by a feed-forward neural network). The bipedal walker is taken from the environment [BipedalWalker-V2](https://gym.openai.com/envs/BipedalWalker-v2) of the OpenAI gym but to vary the terrains, we extend the [work of UberLabs](https://eng.uber.com/poet-open-ended-deep-learning/).

## Code
The code philosophy is simple: we allow you to design any environment you'd like, then you can train a locomotion policy (while optimizing for the morphology of the agent) using augmented random search. Our code saves your models automatically at eppoch intervals and you may set the debug option to True to visualize the progress of your walker at intervals. Alternatively, you can load a saved model with the debug option to visualze what policy and morphology were learnt. 

Here is an example of an environment that we design. It was impossible to solve at first. But we learn bith a walking policy and a body shape that makes the environment solvable.

![Cool morph adaptation](https://github.com/jaks19/bipedal-walker-adaptable-morphology/blob/master/gifs/gif_fail.png)

### To design any custom environment:
Modify the ENV_CONFIG object placed at the top of main.py.

```
ENV_CONFIG = Env_config(
    name='rough',
    ground_roughness=0,
    pit_gap=[1,2],
    stump_width=None,
    stump_height=None,
    stump_float=None,
    stair_height=None,
    stair_width=None,
    stair_steps=None,
    )
```

### To run training
