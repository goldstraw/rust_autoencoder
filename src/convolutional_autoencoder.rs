use crate::IMAGE_SIZE;
use crate::{
    convolutional_layer::ConvolutionalLayer, fully_connected_layer::FullyConnectedLayer,
    layer::Layer, max_pooling_layer::MaxPoolingLayer,
    upscaling_layer::UpscalingLayer,
};

/// A struct that represents an autoencoder (Autoencoder)
pub struct Autoencoder {
    /// A vector of `Layer` objects representing the layers in the autoencoder
    layers: Vec<Box<dyn Layer>>,
}

impl Autoencoder {
    /// Creates a new `Autoencoder` object with an empty vector of layers
    pub fn new() -> Autoencoder {
        let layers: Vec<Box<dyn Layer>> = Vec::new();

        Autoencoder {
            layers,
        }
    }

    /// Adds a convolutional layer to the neural network.
    pub fn add_conv_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        sigmoid: bool,
    ) {
        // Create a new convolutional layer with the specified parameters.
        let conv_layer: ConvolutionalLayer =
            ConvolutionalLayer::new(input_size, input_depth, num_filters, kernel_size, stride, padding, sigmoid);
        let conv_layer_ptr = Box::new(conv_layer) as Box<dyn Layer>;
        // Push the layer onto the list of layers in the neural network.
        self.layers.push(conv_layer_ptr);
    }

    /// Adds a max pooling layer to the neural network
    pub fn add_mxpl_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
    ) {
        // Create a new max pooling layer with the specified parameters
        let mxpl_layer: MaxPoolingLayer =
            MaxPoolingLayer::new(input_size, input_depth);
        let mxpl_layer_ptr = Box::new(mxpl_layer) as Box<dyn Layer>;
        // Push the layer onto the list of layers in the neural network.
        self.layers.push(mxpl_layer_ptr);
    }

    /// Adds a fully connected layer to the neural network
    pub fn add_fcl_layer(&mut self, input_width: usize, input_depth: usize, output_width: usize, output_depth: usize) {
        // Create a new fully connected layer with the specified parameters
        let fcl_layer: FullyConnectedLayer =
            FullyConnectedLayer::new(input_width, input_depth, output_width, output_depth);
        let fcl_layer_ptr = Box::new(fcl_layer) as Box<dyn Layer>;
        // Push the layer onto the list of layers in the neural network.
        self.layers.push(fcl_layer_ptr);
    }

    /// Adds an upscaling layer to the neural network
    pub fn add_upscaling_layer(
        &mut self,
        input_size: usize,
        input_depth: usize,
    ) {
        // Create a new upscaling pooling layer with the specified parameters
        let upscale_layer: UpscalingLayer =
            UpscalingLayer::new(input_size, input_depth);
        let upscale_layer_ptr = Box::new(upscale_layer) as Box<dyn Layer>;
        // Push the layer onto the list of layers in the neural network.
        self.layers.push(upscale_layer_ptr);
    }

    /// Forward propagates an input matrix through the autoencoder.
    pub fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        let mut output: Vec<Vec<Vec<f32>>> = input;

        // Forward propagate through each layer of the network
        for i in 0..self.layers.len() {
            output = self.layers[i].forward_propagate(output);
        }

        // Flatten and return the output of the final layer
        output.clone()
    }

    /// Calculates the error of the last layer of the network
    pub fn last_layer_error(&mut self, image: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        let mut error: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; IMAGE_SIZE]; IMAGE_SIZE]; 3];
        let last_index: usize = self.layers.len() - 1;

        // Calculate the error for each output neuron
        let z = 0; // Grey scale image, so only one channel
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let output: f32 = self.layers[last_index].get_output((z,y,x));
                error[z][y][x] = 2.0 * (output - image[z][y][x]) / (IMAGE_SIZE * IMAGE_SIZE) as f32;
            }
        }

        error
    }

    /// Calculates the cost of the last layer of the network
    pub fn cost(&mut self, image: Vec<Vec<Vec<f32>>>) -> f32 {
        let mut cost: f32 = 0.0;
        let last_index: usize = self.layers.len() - 1;

        // Calculate the error for each output neuron
        let z = 0; // Grey scale image, so only one channel
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let output: f32 = self.layers[last_index].get_output((z,y,x));
                cost += (output - image[z][y][x]).powi(2) / (IMAGE_SIZE * IMAGE_SIZE) as f32;
            }
        }

        cost
    }

    /// Backpropagate the error from the output layer to the input layer
    pub fn back_propagate(&mut self, image: Vec<Vec<Vec<f32>>>) {
        // Retrieve the last layer error to backpropagate
        let mut error: Vec<Vec<Vec<f32>>> = self.last_layer_error(image);

        // Iterate backwards through the layers and backpropagate the error
        for i in (0..self.layers.len()).rev() {
            error = self.layers[i].back_propagate(error);
        }
    }

    /// Update the network with the stored minibatch changes
    pub fn update_layers(&mut self, minibatch_size: usize) {
        // Iterate through the layers and update layers
        for i in 0..self.layers.len() {
            self.layers[i].update_layer(minibatch_size);
        }
    }
}