# delete logs
rm -rf logs

# tell user
echo "DELETED OLD LOGS!"

# run training AND tensorboard at same time
python3 /code/tensorboard_test.py & tensorboard --logdir logs/fit --port=80 --host=0.0.0.0
