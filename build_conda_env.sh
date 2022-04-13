# Run below command to build env:
#   source build_conda_env.sh <YOUR ENV NAME>
# 
# See your current envs with:
#   conda info --envs

if [ $# -eq 0 ]
then
    echo "[INFO]: Pass name of server environment as arg 1 e.g. sh build_conda_env.sh SERVER_ENV"
else
    echo "[SUCCESS]: Building environment..."

    # Create new conda environment
    conda create -y -n $1 python=3.8.10 ipython

    # Activate new environment
    conda activate $1

    # MUST BE PYTHON 3.8.10 TO BE COMPATIBLE WITH SERVERS!
    conda install -y -c conda-forge pyvirtualdisplay
    conda install -y -c conda-forge tensorflow
    conda install -y -c conda-forge keras
    conda install -y -c conda-forge opencv
    conda install -y -c conda-forge gym
    conda install -y -c conda-forge pyglet
    conda install -y -c cogsci pygame
    conda install -y -c conda-forge box2d-py

    # remove env with:
    #   conda env remove -n $1

    echo "[SUCCESS]: Finished!!"
fi 
