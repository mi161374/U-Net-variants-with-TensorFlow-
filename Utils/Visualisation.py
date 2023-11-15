from matplotlib.colors import ListedColormap
import matplotlib.pylab as plt

def display(images, 
            masks, 
            display_range,
            titles=['Ground Truth', 'Prediction'],
            colors=['blue', 'cyan', 'magenta', 'green', 'yellow', 'orange', 'purple', 'red', 'pink', 'black'], 
            figsize=(30, 30)):
  
  assert isinstance(display_range, list), 'display_range must be a list [start, stop, step]'
  assert display_range[1] <= len(images), 'n_display must be <= len(images)'
  assert isinstance(masks, list), 'masks must be a list'

  cmap = ListedColormap(colors)

  for i in range(display_range[0], display_range[1], display_range[2]):
    plt.figure(figsize=figsize)

    for j in range(len(masks)):
      plt.subplot(1,len(masks),j+1)
      plt.title(titles[j]+'  '+str(i+1))
      plt.xticks([])
      plt.yticks([])
      plt.imshow(images[i], cmap='gray')
      chosen_mask = masks[j]
      plt.imshow(chosen_mask[i], alpha=0.5, cmap=cmap)
    plt.show()