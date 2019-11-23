# bipedal-walker-adaptable-morphology

We investigate whether allowing an agent to modify its own morphology has any beneficial effect on the task finding optimal locomotive policies, on terrains of varying difficulty. Work is inspired by David Ha's and UberLabs' previous work using this agent and the flexibility of the terrain in this environment respectively.

We use the augmented random search algorithm to optimize the policy of a bipedal walker (parametrized by a feed-forward neural network). The bipedal walker is taken from the environment [BipedalWalker-V2](https://gym.openai.com/envs/BipedalWalker-v2) of the OpenAI gym but to vary the terrains, we extend the [work of UberLabs](https://eng.uber.com/poet-open-ended-deep-learning/).

## Code
The code philosophy is simple: we allow you to design any environment you'd like, then you can train a locomotion policy (while optimizing for the morphology of the agent) using augmented random search. Our code saves your models automatically at epoch intervals and you may set the debug option to True to visualize the progress of your walker at intervals. Alternatively, you can load a saved model with the debug option to visualize what policy and morphology were learnt. 

Here is an example of an environment that we design. It was impossible to solve at first. But we learn both a walking policy and a body shape that makes the environment solvable.

![Cool morph adaptation](https://github.com/jaks19/bipedal-walker-adaptable-morphology/blob/master/gifs/gif_fail.gif)

![Cool morph adaptation](https://github.com/jaks19/bipedal-walker-adaptable-morphology/blob/master/gifs/gif_pass.gif)

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

### To run training on a chosen environment
python main.py

Here are the args with help on how to use them
```
parser.add_argument('--log_dir', type=str, help='log directory')
parser.add_argument('--num_cores', type=int, help='num cpu cores')
parser.add_argument('--npop', type=int, help='num model replicas per epoch')
parser.add_argument('--num_workers', type=int, help='number of data-collecting workers (will split npop among workers')
parser.add_argument('--saved_model', type=str, default=None, help='saved model path if desired')

parser.add_argument('--sigma', type=float, default=0.1, help='for random search')
parser.add_argument('--alpha', type=float, default=0.03, help='for random search')

parser.add_argument('--scale_limit_lower', type=float, default=1, help='size limit of agent (min)')
parser.add_argument('--scale_limit_upper', type=float, default=1, help='size limit of agent (max)')

parser.add_argument('--debug', type=bool, default=True, help='include with True if want to visualize agent in env')
parser.add_argument('--save_interval', type=int, default=10,  help='save policy after every --this number-- of epochs')
```

Example:
python main.py --log_dir ./logs/ --num_cores 4 --npop 10 --num_workers 3 --save_interval 10 --debug True
