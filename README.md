# Compositional Pattern Producing Networks in Pytorch
Generate high resolution, intensely stimulating images with random neural networks.

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

![alt text](https://github.com/jbial/cppn-pytorch/blob/master/images/tanhtanh_2.png)

Generating Gifs: `python generate.py --frames=10 --name="foo.gif"`

![alt text](https://github.com/jbial/cppn-pytorch/blob/master/gifs/tanhtanh.gif)

Disclaimer: It's in Tensorflow right now, but I will port everything to pytorch soon

## Experiments
* I found the `--scale` and `--z-dim` parameter to have the most impact of visual effects.
* In general, the scale acts as a "zooming" parameter the image space, and the z dimension acts a frequency parameter of the image features

<p float="left">
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/tanhtanh_3.png" width="256" />
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/tanhtanh_2.png" width="256" /> 
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/tanhtanh_2.png" width="256" />
</p>

## Acknowledgement

This project was adapted from [David Ha's amazing blog post](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/)

