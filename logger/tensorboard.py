import matplotlib.pyplot as plt
import numpy as np

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(images, preds, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(len(preds)):
        ax = fig.add_subplot(len(preds)//4 if len(preds)%4==0 else len(preds)//4+1,
                             4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(f"(pred: {preds[idx]}) \n(label: {labels[idx]})",
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig