import matplotlib.pyplot as plt
import os
import numpy as np

def draw(record, path):
    """
    This function will draw the figure of record.
    Record: a dict of list
    Path: output path
    """
    plt.ioff()
    for (k, i) in record.items():
        plt.plot(range(len(i)), i)
        plt.title(k)
        plt.xlim(0, len(i))
        plt.ylim(0, max(i) * 1.2)
        plt.savefig(os.path.join(path, '{}.jpg'.format(k)), dpi=200)
        plt.close()

if __name__ == '__main__':
    test = {'z_loss': [1117.9861326197845, 625.433622853121, 380.89626669539706, 244.70078284593455, 164.09081832425377], 'recon_loss': [0.2274277932423924, 0.22695989221999047, 0.22728739639114592, 0.22773424260705957, 0.22786100674624207], 'total_loss': [1118.2135604130267, 625.6605827453411, 381.1235540917882, 244.9285170885416, 164.318679331]}

    draw(test, '../saved')