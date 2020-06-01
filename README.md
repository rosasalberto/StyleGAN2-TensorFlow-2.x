# StyleGAN2-TensorFlow-2.x
Unofficial implementation of StyleGAN2 *config-f* using TensorFlow 2.x. <br/>  <br/>
Official paper from Nvidia: https://arxiv.org/abs/1912.04958 <br/>
Official repo using TensorFlow 1.x: https://github.com/NVlabs/stylegan2

![Stylegan2 random walk in latent space](images/ffhq_latent.gif)

## Extra features

* Support for TensorFlow custom operations
* Support for CPU usage

The Conv2D op currently does not support grouped convolutions on the CPU. In consequence, when running with CPU, batch size should be 1.

Download network parameters to *weights* folder manually https://drive.google.com/drive/folders/1rhuvN90EGsRhvjQq5gio8VYw7f0LojaK?usp=sharing, or simpy run *download.py* script located in *weights* folder.

```
# Create stylegan2 architecture (generator and discriminator) using cuda operations.
model = StyleGan2(resolution, impl='cuda', gpu=True)

# Load stylegan2 'ffhq' (generator and discriminator) using tensorflow operations.
model = StyleGan2(weights='ffhq', impl='ref', gpu=True)

# Load stylegan2 'horse' (generator and discriminator) using tensorflow operations in cpu.
model = StyleGan2(weights='horse', impl='ref', gpu=False)

# Load only generator network with 'car' weights
generator = StyleGan2Generator(weights='car', impl, gpu)
```

Examples on how to create, load and run the networks can be found in *example_how_to_use* notebook.<br/>
Examples on how to make a random walk in the latent vector and generate a gif, can be found in *example_latent_changes* notebook.

Training loop and metrics has not been implemented yet. Stay tuned.
