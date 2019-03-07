import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from ggplot import *

def compare_images(VariationalAutoEncoder,
                   test_loader,
                   number_original_images=5,
                   number_generated_images=5):

        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        images_index = np.random.choice(range(example_data.shape[0]), number_original_images)
        fig = plt.figure()

        subfigure = 1
        for i in range(number_original_images):

            index_image = images_index[i]
            image = example_data[index_image][0]

            plt.subplot(number_original_images, number_generated_images + 1, subfigure)
            plt.title("Ground Truth: {}".format(example_targets[index_image]))

            plt.imshow(image.detach().numpy(), cmap='gray', interpolation='none')

            plt.xticks([])
            plt.yticks([])

            subfigure += 1

            for _ in range(number_generated_images):

                plt.subplot(number_original_images, number_generated_images + 1, subfigure)

                image_reconstructed = VariationalAutoEncoder.reconstruct_img(image).detach().numpy().reshape(28,28)
                plt.title("Generated image")
                plt.imshow(image_reconstructed, cmap='gray', interpolation='none')

                plt.xticks([])
                plt.yticks([])

                subfigure += 1

        plt.show()


def t_SNE(test_loader, variational_autoencoder):
    """
    single batch
    :param test_loader:
    :param variational_autoencoder:
    :return:
    """
    features = list()
    labels = list()

    examples = enumerate(test_loader)

    for batch_idx, (example_data, example_targets) in examples:
        for idx in range(example_data.shape[0]):

            x = example_data[idx][0]
            z_loc, z_scale = variational_autoencoder.encoder(x)

            values = np.concatenate((z_loc[0].detach().numpy(), z_scale[0].detach().numpy()))
            features.append(values)
            label = example_targets[idx].detach().numpy()
            labels.append(label)

    loc_cols = ['loc_' + str(i) for i in range(variational_autoencoder.z_dim)]
    scale_cols = ['scale_' + str(i) for i in range(variational_autoencoder.z_dim)]
    feat_cols = loc_cols + scale_cols

    df = pd.DataFrame(features, columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))


    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df_tsne = df.copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]

    chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
            + geom_point(size=70, alpha=0.8) \
            + ggtitle("tSNE dimensions colored by digit")

    chart.show()






