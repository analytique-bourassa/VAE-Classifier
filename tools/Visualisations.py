import matplotlib.pyplot as plt
import numpy as np

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

