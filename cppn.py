import tensorflow as tf
import numpy as np

##########################################
## Experiment with CPPN's 
##########################################

class CPPN: 
    """Randomly initialized generator network
    """    

    def __init__(self, im_size, batch_size=1, n_dim=32, z_dim=32, layers=3, channels=1, scale=1.0):
        self.im_size = im_size
        self.batch_size = batch_size
        self.n_dim, self.z_dim = n_dim, z_dim
        self.layers = layers
        self.channels = channels
        self.scale = scale
        self.num_pixels = self.im_size ** 2

        # Create the placeholders
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.x = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='x')
        self.y = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='y')
        self.r = tf.placeholder(tf.float32, [self.batch_size, None, 1], name='r')

        # Create the layers
        self.dense_z = self._dense(self.n_dim, 'z') 
        self.dense_x = self._dense(self.n_dim, 'x', bias=False)
        self.dense_y = self._dense(self.n_dim, 'y', bias=False) 
        self.dense_r = self._dense(self.n_dim, 'r', bias=False)

        self.generator = self._generate_images()
        self.init()

    def _generate_images(self):
        """Foward pass to produce an image
        """        

        # Preprocessing layer
        z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
        self.scale * tf.ones([self.num_pixels, 1], dtype=tf.float32)
        z_input = tf.reshape(z_scaled, [self.batch_size * self.num_pixels, self.z_dim])
        x_input = tf.reshape(self.x, [self.batch_size * self.num_pixels, 1])
        y_input = tf.reshape(self.y, [self.batch_size * self.num_pixels, 1])
        r_input = tf.reshape(self.r, [self.batch_size * self.num_pixels, 1])

        # Forward pass
        h = tf.nn.softplus(self.dense_x(x_input) + self.dense_y(y_input) + self.dense_z(z_input) + self.dense_r(r_input))
        for i in range(self.layers):
            h = tf.tanh(self._dense(self.n_dim, f'h_{i}')(h))
        h = tf.nn.sigmoid(self._dense(self.channels, 'out')(h)) 
        return tf.reshape(h, [self.batch_size, self.im_size, self.im_size, self.channels])

    def _dense(self, units, name, bias=True):
        """Get a tensorflow dense layer 
        """    
        def func(x):
            return tf.layers.dense(x, units, kernel_initializer=tf.random_normal_initializer, 
                        bias_initializer=tf.random_normal_initializer, use_bias=bias, name=name, reuse=tf.AUTO_REUSE)
        return func

    def _get_coords(self):
        """Returns matrices of the x, and y coordinates, and radii
        """
        size = 2 * self.scale * (np.arange(self.im_size) - (self.im_size - 1)/2.0) / (self.im_size - 1)
        x = np.tile(size, (self.im_size, 1))
        y = np.tile(size.reshape(self.im_size, 1), (1, self.im_size))
        r = np.sqrt(x*x + y*y)
        xmat = np.tile(x.flatten(), self.batch_size).reshape(self.batch_size,-1,1)
        ymat = np.tile(y.flatten(), self.batch_size).reshape(self.batch_size,-1,1)
        rmat = np.tile(r.flatten(), self.batch_size).reshape(self.batch_size,-1,1)
        return xmat, ymat, rmat

    def init(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, latent_code=None):
        if latent_code is None:
            latent_code = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)

        # Get the generator and coordinates/radii
        generator = self._generate_images()
        x, y, r = self._get_coords()

        images = self.sess.run(generator, feed_dict={self.z: latent_code, self.x: x, self.y: y, self.r: r})
        return images.squeeze()

