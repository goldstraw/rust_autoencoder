# Rust Autoencoder from Scratch

This repository contains a Rust implementation of a convolutional autoencoder built from scratch. The autoencoder is designed to differentiate between handwritten digits of the number one and all other data.

## Overview

The repository contains the following main components:

```
src/
├── convolutional_autoencoder.rs
├── convolutional_layer.rs
├── fully_connected_layer.rs
├── layer.rs
├── lib.rs
├── main.rs
├── max_pooling_layer.rs
├── run.rs
└── upscaling_layer.rs
```

* `convolutional_autoencoder.rs`: Defines the structure of the autoencoder model.
* `convolutional_layer.rs`: Implements the convolutional layer for the autoencoder.
* `fully_connected_layer.rs`: Implements the fully connected layer for the autoencoder.
* `layer.rs`: Defines the interface for the autoencoder layers.
* `lib.rs`: The Rust library file.
* `main.rs`: A demo of the autoencoder's use.
* `max_pooling_layer.rs`: Implements the max pooling layer for the autoencoder.
* `run.rs`: Contains functions to run the autoencoder.
* `upscaling_layer.rs`: Implements the upscaling layer for the autoencoder.

## Installation

To use this CNN implementation, you must have Rust and Cargo installed on your machine. After installing Rust and Cargo, you can clone this repository to your local machine and build the project with the following command:

```
$ cargo build
```

## Usage

To run the demo of the CNN, add the MNIST dataset to the repository in a folder called `data` and use the following command:

```
$ cargo run --release
```

This command will run a demo of the CNN and train it on the MNIST dataset.

## Further Reading

For more information about this project, read [my blog post on autoencoders](https://charliegoldstraw.com/articles/autoencoder/):

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the `LICENSE` file for details.
