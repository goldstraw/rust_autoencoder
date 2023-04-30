use crate::{LEARNING_RATE, layer::Layer};
use rand_distr::{Distribution, Normal};

/// Calculates the sigmoid of a given input value
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Function to calculate inverse derivative of sigmoid of a given input value
fn inv_deriv_sigmoid(x: f32) -> f32 {
    let z: f32 = (x / (1.0 - x)).ln(); // Undoes sigmoid activation
    sigmoid(z) * (1.0 - sigmoid(z)) // Returns derivative of sigmoid of the weighted sum
}

/// Defines a `ConvolutionalLayer` structure.
pub struct ConvolutionalLayer {
    input_size: usize,                       // Input size
    input_depth: usize,                      // Input depth
    num_filters: usize,                      // Number of filters
    kernel_size: usize,                      // Kernel size
    output_size: usize,                      // Output size
    stride: usize,                           // Stride
    padding: usize,                          // Padding
    biases: Vec<f32>,                        // Vector of biases
    bias_changes: Vec<f32>,                  // Bias minibatch changes
    kernels: Vec<Vec<Vec<Vec<f32>>>>,        // Kernel values
    kernel_changes: Vec<Vec<Vec<Vec<f32>>>>, // Kernel minibatch changes
    sigmoid: bool,                           // Activation Function
    input: Vec<Vec<Vec<f32>>>,               // Input layer
    output: Vec<Vec<Vec<f32>>>,          // Output layer
}

impl ConvolutionalLayer {
    /// Creates a new convolutional layer with the given parameters
    pub fn new(
        input_size: usize,
        input_depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        sigmoid: bool,
    ) -> ConvolutionalLayer {
        // Initialize the biases and kernels with empty vectors
        let mut biases = vec![];
        let bias_changes = vec![0.0; num_filters];
        let mut kernels = vec![vec![vec![vec![]; kernel_size]; input_depth]; num_filters];
        let kernel_changes = vec![vec![vec![vec![0.0; kernel_size]; kernel_size]; input_depth]; num_filters];

        // Use He initialisation by using a mean of 0.0 and a standard deviation of sqrt(2/(input_channels * num_params))
        let normal = Normal::new(0.0, (2.0/(input_depth*kernel_size.pow(2)) as f32).sqrt()).unwrap();

        // Fill the biases and kernels with random values from the normal distribution
        for f in 0..num_filters {
            biases.push(0.1);
            for i in 0..input_depth {
                for j in 0..kernel_size {
                    for _ in 0..kernel_size {
                        kernels[f][i][j].push(normal.sample(&mut rand::thread_rng()));
                    }
                }
            }
        }

        let output_size: usize = ((input_size - kernel_size) / stride) + 1 + 2 * padding;

        // Create the ConvolutionalLayer struct and return it
        let layer: ConvolutionalLayer = ConvolutionalLayer {
            input_size,
            input_depth,
            num_filters,
            kernel_size,
            output_size,
            stride,
            padding,
            biases,
            bias_changes,
            kernels,
            kernel_changes,
            sigmoid,
            input: vec![],
            output: vec![vec![vec![0.0; output_size]; output_size]; num_filters],
        };

        layer
    }

    /// Pads a 3D matrix with zeroes.
    fn pad(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        let padded_size = self.input_size + 2 * self.padding;
        let mut padded = vec![vec![vec![0.0; padded_size]; padded_size]; self.input_depth];

        for d in 0..self.input_depth {
            for h in self.padding..(padded_size - self.padding) {
                for w in self.padding..(padded_size - self.padding) {
                    padded[d][h][w] = input[d][h - self.padding][w - self.padding];
                }
            }
        }

        padded
    }
}

impl Layer for ConvolutionalLayer {
    /// Forward propagates the input data through the Convolutional layer.
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Store the input data in a member variable for future reference.
        self.input = input.clone();
        let padded_input = self.pad(self.input.clone());

        // Iterate through each output point in the output matrix.
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                // Calculate the starting point for the convolutional kernel.
                let left = x * self.stride;
                let top = y * self.stride;
                // Iterate through each filter in the network.
                for f in 0..self.num_filters {
                    // Initialize the output value with the bias value for the filter.
                    self.output[f][y][x] = self.biases[f];

                    // Iterate through each input channel.
                    for f_i in 0..self.input_depth {
                        for y_k in 0..self.kernel_size {
                            for x_k in 0..self.kernel_size {
                                // Retrieve the value of the input data at the current location.

                                let val: f32 = padded_input[f_i][top + y_k][left + x_k];

                                // Update the output value with the result of the convolution.
                                self.output[f][y][x] += self.kernels[f][f_i][y_k][x_k] * val;
                            }
                        }
                    }
                }
            }
        }

        for f in 0..self.num_filters {
            for y in 0..self.output_size {
                for x in 0..self.output_size {
                    if self.sigmoid {
                        self.output[f][y][x] = sigmoid(self.output[f][y][x]);
                    } else {
                        self.output[f][y][x] = self.output[f][y][x].max(0.0);
                    }
                }
            }
        }

        self.output.clone()
    }

    /// Back propagates the error through the convolutional layer and updates the kernel and biases.
    /// Returns the previous layer's error.
    fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Initialize previous error vector
        let mut prev_error: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![0.0; self.input_size]; self.input_size]; self.input_depth];

        // Iterate through each output element
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                // Calculate receptive field coordinates
                let left = x * self.stride;
                let top = y * self.stride;
                // Iterate through each filter
                for f in 0..self.num_filters {
                    // Handle sigmoid activation function for last layer
                    if self.sigmoid {
                        // Calculate the derivative of the sigmoid for the current output
                        let derivative = inv_deriv_sigmoid(self.output[f][y][x]);
                        // Update the bias
                        self.bias_changes[f] -= error[f][y][x] * derivative * LEARNING_RATE;
                        for y_k in 0..self.kernel_size {
                            for x_k in 0..self.kernel_size {
                                for f_i in 0..self.input_depth {
                                    // Update previous error and kernel weights
                                    if top + y_k >= self.padding
                                            && top + y_k < self.input_size + self.padding
                                            && left + x_k >= self.padding
                                            && left + x_k < self.input_size + self.padding {
                                        prev_error[f_i][top+y_k-self.padding][left+x_k-self.padding] +=
                                            self.kernels[f][f_i][y_k][x_k] * error[f][y][x] * derivative;
                                        self.kernel_changes[f][f_i][y_k][x_k] -=
                                            self.input[f_i][top+y_k-self.padding][left+x_k-self.padding] * error[f][y][x] * derivative * LEARNING_RATE;
                                    } 
                                }
                            }
                        }
                    // Only affect neurons which had an impact on the error if ReLU is used
                    } else if self.output[f][y][x] > 0.0 {
                        // Update the bias
                        self.bias_changes[f] -= error[f][y][x] * LEARNING_RATE;
                        for y_k in 0..self.kernel_size {
                            for x_k in 0..self.kernel_size {
                                for f_i in 0..self.input_depth {
                                    // Update previous error and kernel weights
                                    if top + y_k >= self.padding
                                            && top + y_k < self.input_size + self.padding
                                            && left + x_k >= self.padding
                                            && left + x_k < self.input_size + self.padding {
                                        prev_error[f_i][top+y_k-self.padding][left+x_k-self.padding] +=
                                            self.kernels[f][f_i][y_k][x_k] * error[f][y][x];
                                        self.kernel_changes[f][f_i][y_k][x_k] -= self.input[f_i][top+y_k-self.padding][left+x_k-self.padding]
                                            * error[f][y][x]
                                            * LEARNING_RATE;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return previous layer's error
        prev_error
    }

    /// Returns the output at an index of the fully connected layer.
    fn get_output(&mut self, index: (usize, usize, usize)) -> f32 {
        self.output[index.0][index.1][index.2]
    }

    fn update_layer(&mut self, minibatch_size: usize) {
        for f in 0..self.num_filters {
            self.biases[f] += self.bias_changes[f] / minibatch_size as f32;
            self.bias_changes[f] = 0.0;
            for i in 0..self.input_depth {
                for j in 0..self.kernel_size {
                    for k in 0..self.kernel_size {
                        self.kernels[f][i][j][k] += self.kernel_changes[f][i][j][k] / minibatch_size as f32;
                        self.kernel_changes[f][i][j][k] = 0.0;
                    }
                }
            }
        }
    }
}