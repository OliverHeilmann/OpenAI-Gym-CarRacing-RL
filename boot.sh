# Run Docker Container with (use your details and choose GPU):
# 	hare run -p 10000:80 --gpus device=0 --rm -v "$(pwd)":/code oah33/docker_rl

# delete logs
rm -rf logs

# tell user
echo "DELETED OLD LOGS!"

# run training AND tensorboard at same time
python3 /code/tensorboard_test.py & tensorboard --logdir logs/fit --port=80 --host=0.0.0.0
