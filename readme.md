# Code Description
WIP

# Server Docker Image
```text
hare run --rm -v "$(pwd)":/<file dir> <docker image> python3 /<file dir>/<filename>.py
```

e.g. 
```text
hare run --rm -v "$(pwd)":/code oah33/docker_rl python3 /code/DQN.py
```


# Contributors
1) Oliver Heilmann: oah33@bath.ac.uk
2) Sam Barba: sb3193@bath.ac.uk
3) Isaac Anako: iaa57@bath.ac.uk

# Notes
1) To select interpreter in VS Code--> cmd+shift+p, Python. Select Interpreter, CHOOSE PREFERENCE
