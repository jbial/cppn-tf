import tensorflow as tf
import numpy as np
import visualize as vis 
import argparse
from cppn import CPPN

##########################################
## Generate images,and gifs
##########################################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--im-size", type=int, default=512, help="Size of the image.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--units", type=int, default=32, help="Number of units per layer.")
    parser.add_argument("--z-dim", type=int, default=32, help="Size of latent vector.")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers.")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels in output images.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for the images.")
    parser.add_argument("--name", type=str, default=None, help="Name of image for saving. Default to None for no saving." )
    parser.add_argument("--frames", type=int, default=None, help="Number of frames for the gif.")
    parser.add_argument("--scale-list", type=str, default=None, help="Comma-separated list of scales for the images.")
    parser.add_argument("--display-cols", type=int, default=4, help="Number of columns for showing image batches.")
    parser.add_argument("--same-z", action='store_true', default=False, help="Use the same latent vector for all param lists.")
    args = parser.parse_args()
                    
    # Initialize the model
    model = CPPN(im_size=args.im_size, batch_size=args.batch_size, units=args.units, z_dim=args.z_dim,
                layers=args.layers, channels=args.channels, scale=args.scale)

    # Save gif if specified
    if args.frames and args.name:
        vis.generate_gif(model, frames=args.frames, size=args.z_dim, name=args.name)
        print("Saved gif as {}".format(args.name))
        raise SystemExit

    # Get the scale parameter list
    scales = [float(s) for s in args.scale_list.split(',')] if args.scale_list is not None else None

    # generate images
    z = model.get_code() if args.same_z else None
    images = [model(
        scale_param=s,
        latent_code=z
    ) for s in scales] if scales else model()
    
    # Show and save images
    if args.batch_size > 1 or len(scales) > 1:
        vis.show_images(images, args.display_cols) 
        if args.name: vis.save_images(images, args.name)
    else:
        vis.show_image(images)
        if args.name: vis.save_image(images, args.name)

if __name__ == '__main__': main()
