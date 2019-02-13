import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from PIL import Image

#######################################
## Methods for CPPN visualization
#######################################

def get_zs(size, distribution):
    """Get a randomly sampled latent vector
    """
    dist_dict = dict([('normal', np.random.normal), ('uniform', np.random.uniform)])
    return dist_dict[distribution](size=(2, 1, size))

def show_image(image):
    """Show a single image
    """
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()

def show_images(images, num_images):
    """Show sampled images
    """
    fig, axes = plt.subplots(num_images, num_images)
    indices = np.random.choice(images.shape[0], 
                    (num_images, num_images), replace=False)
    for i, row_inds in zip(axes, indices):
            for j, inds in zip(i, row_inds):
                j.set_axis_off()
                j.imshow(images[inds])
    plt.show()

def save_image(image, name, path='images/'):
    """Save generated images
    """
    im = np.uint8(255 * image)
    im = Image.fromarray(im)
    im.save(path + name)

def save_images(images, name):
    """Save a batch of generated images
    """
    for i, im in enumerate(images):
        file_name = name.split('.')
        save_image(im, "{0}_{1}.{2}".format(file_name[0], i+1, file_name[1])) 

def to_image(image):
    """Converts numpy to PIL
    """
    return Image.fromarray(np.uint8(255 * image.squeeze()))

def clear_image_dir():
    """Empty the gif image directory to produce a new gif
    """
    for f in glob.glob('gifs/gif_imgs/*'):
        os.remove(f)

def generate_gif(model, frames, size, name, distribution='normal'):
    """Generate an interpolation gif
    """

    # Get interpolation information
    z1, z2 = get_zs(size, distribution)
    coeff = np.linspace(0, 1, frames+2)
    interp = lambda a: (1-a) * z1 + a * z2    

    # Store the interpolated images
    images = [to_image(model(latent_code=interp(l))) for l in coeff]
    
    # append the reverse sequence
    images += images[::-1][1:]
    durations = [0.5] + frames*[0.1] + [1.0] + frames*[0.1] + [0.5]
        
    # Write to GIF
    print("Writing images to gif")
    images[0].save('./gifs/' + name, save_all=True, append_images=images[1:],
                    duration=durations, loop=0) 
