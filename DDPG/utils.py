import numpy as np
import matplotlib.pyplot as plt
import os

# This noise generator was taken from a source
class NoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))

        return self.x

def preprocess(img, greyscale=False):
    img = img.copy()
    # Remove numbers and enlarge speed bar
    for i in range(88, 93 + 1):
        img[i, 0:12, :] = img[i, 12, :]

    # Unify grass color
    replace_color(img, original=(102, 229, 102), new_value=(102, 204, 102))

    if greyscale:
        img = img.mean(axis=2)
        img = np.expand_dims(img, 2)

    # Make car black
    car_color = 68.0
    car_area = img[67:77, 42:53]
    car_area[car_area == car_color] = 0

    # Scale from 0 to 1
    img = img / img.max()
    # Unify track color
    img[(img > 0.411) & (img < 0.412)] = 0.4
    img[(img > 0.419) & (img < 0.420)] = 0.4
    # Change color of kerbs
    game_screen = img[0:83, :]
    game_screen[game_screen == 1] = 0.80
    return img

def prepend_tuple(new_dim, some_shape):
    some_shape_list = list(some_shape)
    some_shape_list.insert(0, new_dim)
    return tuple(some_shape_list)

def replace_color(data, original, new_value):
    r1, g1, b1 = original
    r2, g2, b2 = new_value

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 rewards')
    plt.savefig(figure_file)

def save_result_to_csv(name, data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(f"{path}" + name + ".csv", data, delimiter=",")
