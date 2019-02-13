# Compositional Pattern Producing Networks in Pytorch
Generate intensely stimulating images and gifs with random neural networks adapted from David Ha's blog post ().

## Arguments
`python generate.py` takes as arguments:
* `--im-size`: Size of the image.
* `--batch-size`: Number of images to generate.
* `--n-dim`: Number of units per layer.
* `--z-dim`: Size of input latent vector.
* `--layers`: Number of layers in the network.
* `--channels`: Number of channels in output images.
* `--scale`: Scaling factor for the images.
* `--name`: Name of image for saving. Default is None, specify for saving.
* `--frames`: Number of frames for the gif. Default is None, specify an integer to save a gif file.

## Usage
Generating Images: `python generate.py` 

![alt text]()

Generating Gifs: `python generate.py --frames=10 --name="foo.gif"`


Disclaimer: It's in Tensorflow right now, but I will port everything to pytorch soon
