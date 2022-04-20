# Run Docker Container with (use your details and choose GPU):
# 	hare run -p 10000:80 --gpus device=0 --rm -v "$(pwd)":/code oah33/docker_rl

# delete logs
rm -rf logs

# tell user
echo "DELETED OLD LOGS!"

# run training AND tensorboard at same time
# IMPORTANT! Change /code/<FILENAME.PY> to run your chosen script!
python3 /code/DQN.py & tensorboard --logdir logs/fit --port=80 --host=0.0.0.0