# Compositional Pattern Producing Networks in Tensorflow
Generate high resolution, intensely stimulating images with random neural networks.

## Arguments
`python generate.py` takes as arguments:
* `--im-size`: Size of the image.
* `--batch-size`: Number of images to generate.
* `--units`: Number of units per layer.
* `--z-dim`: Size of input latent vector.
* `--layers`: Number of layers in the network.
* `--channels`: Number of channels in output images.
* `--scale`: Scaling factor for the images.
* `--name`: Name of image for saving. Default is None, specify for saving.
* `--frames`: Number of frames for the gif. Default is None, specify an integer to save a gif file.
* `--scale-list`: Comma-separated list of scales for the images.
* `--display-cols`: Number of columns for showing image batches.
* `--same-z`: Use the same latent vector for all param lists.

## Usage
Generating Images: `python generate.py` 

<p float="left">
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/tanhcos_1.png" width="450" />
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/tanhtanh_1.png" width="450" /> 
 </p>
 <p float='left'>
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/tanhtanh_2.png" width="256" />
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/tanhcos_2.png" width="256" />
</p>

![alt text](https://github.com/jbial/cppn-pytorch/blob/master/images/random.jpg)

Generating Gifs: `python generate.py --frames=10 --name="tanhtanh.gif"`

![alt text](https://github.com/jbial/cppn-pytorch/blob/master/gifs/tanhtanh.gif)


## Experiments
* The `--scale` parameter acts as a "zooming" parameter in the image space.
* The `--z-dim` parameter acts a control parameter on the frequency of generated features.
* Both `--units` and `--layers` control the noise level in the generated images, which makes sense since these parameters adjust the number of weights which dictate the 'representational power' of the neural network. Try setting `--layers` or `--units` to 0 and then 64 for yourself.


The effect of exponentially increasing the scale parameter (all other params default): `--scale-list=1,5,25,125`
<p float="left">
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/scale_1.jpg" width="200" />
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/scale_2.jpg" width="200" /> 
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/scale_3.jpg" width="200" />
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/scale_4.jpg" width="200" />
</p>

The effect of exponentially increasing the z dimension (all other params default): `--z-dim` from {1,8,64,512}
<p float="left">
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/z_1.png" width="200" />
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/z_8.png" width="200" /> 
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/z_64.png" width="200" />
  <img src="https://github.com/jbial/cppn-pytorch/blob/master/images/z_512.png" width="200" />
</p>

## Acknowledgement

This project was adapted from [David Ha's amazing blog post](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/)

