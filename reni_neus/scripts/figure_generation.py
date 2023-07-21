import os
import matplotlib.pyplot as plt
from PIL import Image

def create_teaser_figure(methods):
    # Define names of the images
    img_names = ['img', 'accumulation', 'albedo', 'depth', 'normal']

    # Create subplot for each image and an additional column for ground truth
    fig, axs = plt.subplots(len(methods), len(img_names)+2, figsize=(20, 4*len(methods)))

    for i, (method, info) in enumerate(methods.items()):
        # Define image filenames based on step
        img_files = [f'{name}_{info["step"]}.png' for name in img_names]
        ground_truth_file = f'ground_truth_{info["step"]}.png'

        # Load images
        images = [Image.open(os.path.join(info["base_path"], file)) for file in img_files]
        ground_truth = Image.open(os.path.join(info["base_path"], ground_truth_file))

        if i == 0:  # Only show ground truth for the first method to avoid repetition
            axs[i, 0].imshow(ground_truth)
            axs[i, 0].axis('off')  # No axis for images
            axs[i, 0].set_title('Input Image')
        else:
            axs[i, 0].axis('off')  # No axis for images

        # Second column is for method names
        axs[i, 1].axis('off')  # No axis for text
        axs[i, 1].text(0.5, 0.5, method, horizontalalignment='center',
                       verticalalignment='center', fontsize=12, transform=axs[i, 1].transAxes)

        # Subsequent columns are for images
        for j, img in enumerate(images):
            axs[i, j+2].imshow(img)
            axs[i, j+2].axis('off')  # No axis for images
            axs[i, j+2].set_title(img_names[j])

    # Save the teaser figure
    teaser_fig_name = f'teaser_figure.png'
    plt.savefig(teaser_fig_name, bbox_inches='tight')
    print(f'Teaser figure saved as {teaser_fig_name}.')

# Test the function
# methods_dict = {
#    "method1": {"base_path": "/path/to/your/images/method1", "step": 1},
#    "method2": {"base_path": "/path/to/your/images/method2", "step": 2}
# }
# create_teaser_figure(methods_dict)
