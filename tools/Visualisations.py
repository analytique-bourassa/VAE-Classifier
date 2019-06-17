import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from ggplot import *
from sklearn.metrics import confusion_matrix

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

                if (subfigure - 2) % (number_generated_images + 1) == 0:
                    plt.title("Generated images")

                plt.imshow(image_reconstructed, cmap='gray', interpolation='none')

                plt.xticks([])
                plt.yticks([])

                subfigure += 1

        plt.show()


def t_SNE(test_loader, variational_autoencoder):

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

def describe_statistic_per_label(test_loader, variational_autoencoder):

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

    print(df.groupby("label").describe())


def show_confusion_matrix(test_loader, classifier):

    y_test = list()
    y_pred = list()

    examples = enumerate(test_loader)

    for batch_idx, (example_data, example_targets) in examples:

        outputs = classifier(example_data)

        for idx in range(example_data.shape[0]):

            current_prediction = outputs[idx].detach().numpy()
            y_pred.append(current_prediction)

            label = example_targets[idx].detach().numpy()
            y_test.append(label)

    y_pred = np.array(y_pred)
    y_pred = np.argmax(y_pred,axis=1)

    labels = [str(i) for i in range(10)]

    cm = confusion_matrix(y_test, y_pred)
    # Only use the labels that appear in the data
    classes = labels

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title="normalized confusion matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

