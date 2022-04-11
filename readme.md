# Code Description
WIP

# Server Stuff
```text
hare run --rm -v "$(pwd)":/<file dir> <docker image> python3 /<file dir>/<filename>.py
```

e.g. 
```text
hare run --rm -v "$(pwd)":/code oah33/docker_rl python3 /code/DQN.py
```

Build Docker Image:
```text
hare build -t <username>/<choose docker image name> . 
```

Login to server with:
```text
ssh -L 16006:127.0.0.1:6006 <username>@<server...>
```

Access Tensorboard on port:
```text
tensorboard --logdir logs/fit --port 6006
```


# Contributors
1) Oliver Heilmann: oah33@bath.ac.uk
2) Sam Barba: sb3193@bath.ac.uk
3) Isaac Anako: iaa57@bath.ac.uk

# Notes
1) To select interpreter in VS Code--> cmd+shift+p, Python. Select Interpreter, CHOOSE PREFERENCE
