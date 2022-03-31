# Description of Usage
Rather than using the _gym_ version of the *car_racing* environment, users can use a custom environment instead; note that this environment still uses the _Car_ class from _gym_ (see [car_racing_mod.py](env_mod/car_racing_mod.py) for more details).

Importing local (mod) version instead of gym version:
```python
# Replace:
import gym

# With:
from env_mod.car_racing_mod import CarRacing
```

Create environment object:
```python
# Replace:
env = ENV.make('CarRacing-v0').env

# With:
env = CarRacing()
```