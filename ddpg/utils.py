import numpy as np
import matplotlib.pyplot as plt

# This noise generator was taken from a source
class NoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        if mean.shape != std_dev.shape:
            raise ValueError('Mean shape: {} and std_dev shape: {} should be the same!'.format(
                mean.shape, std_dev.shape))

        # This shape will be generated
        self.x_shape = mean.shape
        self.x = None

        self.reset()

    def reset(self):
        # Reinitialize generator
        self.x = np.zeros_like(self.x_shape)

    def generate(self):
        # The result is based on the old value
        # The second segment will keep values near a mean value
        # It uses normal distribution multiplied by a standard deviation
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.x_shape))

        return self.x


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
