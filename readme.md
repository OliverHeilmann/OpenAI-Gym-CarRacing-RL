# Code Description
WIP

# Server Stuff

## In Code
Open AI Gym uses pyglet which requires a screen/ monitor to function. The servers don't have said screens so we make virtual ones within our code to spoof the code into thinking we do. See below:

```python
import pyvirtualdisplay

# Creates a virtual display for OpenAI gym
pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
```

To use Tensorboard, we need to create the appropriate **logs/fit** event data in the expected format. Add the first two lines at the top of your code and then add the callbacks line into the model.fit function.
```python
# Setup TensorBoard model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

...

self.model.fit(state,action_vals,epochs=1,verbose=0, callbacks=[tensorboard_callback])  # note TensorBoard callback!
```


## In Terminal
Login to the servers using your own credentials. If at any point you need help, go to [HEX](https://hex.cs.bath.ac.uk/) for support, it's a great source!
```text
<!-- login to server as normal via terminal-->
ssh oah33@garlick.cs.bath.ac.uk
```

Only need to do this the first time... We build the Docker container because the server GPUs won't work without it.
```text
<!-- build docker container if not already present -->
hare build -t oah33/docker_rl .
```

Run the Docker container and pass back the relevant information to the server/ host.
```text
<!-- run docker container with python script -->
hare run -p 10000:80 --gpus device=0 --rm -v "$(pwd)":/code oah33/docker_rl python3 /code/DQN.py
```
A lot to unpack in this command, lets break it down:
* ```hare run```: Use docker to run the following inside a virtual machine.
* ```-p 10000:80```: Connect the Docker container port 80 to server host port 10000.
* ```--gpus device=0```: Access GPU number 0 specifically (see Hex for more info on GPU selection).
* ```--rm```: Clean up after the container finishes.
* ```-v "$(pwd)":/app - $(pwd)```: $(pwd) is the directory you are currently in (pwd = print working directory), so it is mounting your current directory in the folder /app inside the container.
* ```python``` - The image to use. python is a pretty basic one (just Python!), so you will generally want one with more libraries.
* ```python3 /app/__main__.py```: The command to run inside the virtual machine. 

In a new terminal, connect with the server and daisy chain the ports from the Docker container (80) to the server (10000) and then to localhost/ your machine (8080). This way, you can access the information being generated through your web browser.
```text
<!-- in new terminal, connect to port from server to localhost -->
ssh -L 8080:localhost:10000 oah33@garlick.cs.bath.ac.uk

<!-- paste the below into a browser to access TensorBoard results -->
localhost:8080
```

If everything has worked properly, you should now see the TensorBoard page in your browser.


## Debugging Server Stuff
```text
jupyter notebook --no-browser --port=10000

tensorboard --logdir logs/fit --port=10000

hare run -p 10000:80 --rm -v "$(pwd)":/code oah33/docker_rl jupyter notebook --allow-root --no-browser --port=80 --ip=0.0.0.0

tensorboard --logdir logs/fit --port 6006

hare me

hare attach <container name>

hare exec <container name> tensorboard --logdir logs/fit --port=80 --host=0.0.0.0

hare run -p 10000:80 --gpus device=0 --rm -v "$(pwd)":/code oah33/docker_rl
hare exec agitated_elion python3 /code/tensorboard_test.py

```

# Contributors
1) Oliver Heilmann: oah33@bath.ac.uk
2) Sam Barba: sb3193@bath.ac.uk
3) Isaac Anako: iaa57@bath.ac.uk

# Notes
1) To select interpreter in VS Code--> cmd+shift+p, Python. Select Interpreter, CHOOSE PREFERENCE
