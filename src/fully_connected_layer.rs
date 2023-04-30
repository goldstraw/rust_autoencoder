use crate::{LEARNING_RATE, layer::Layer};
use rand_distr::{Distribution, Normal};

// /// Calculates the sigmoid of a given input value
// fn sigmoid(x: f32) -> f32 {
//     1.0 / (1.0 + (-x).exp())
// }

// /// Function to calculate inverse derivative of sigmoid of a given input value
// fn inv_deriv_sigmoid(x: f32) -> f32 {
//     let z: f32 = (x / (1.0 - x)).ln(); // Undoes sigmoid activation
//     sigmoid(z) * (1.0 - sigmoid(z)) // Returns derivative of sigmoid of the weighted sum
// }

/// Defines a `FullyConnectedLayer` structure.
pub struct FullyConnectedLayer {
    input_size: usize,              // Number of neurons in input layer.
    input_width: usize,             // Width of input matrix
    input_depth: usize,             // Number of input channels
    output_width: usize,            // Width of output
    output_depth: usize,            // Depth of output
    output_size: usize,             // Number of neurons in output layer.
    weights: Vec<Vec<f32>>,         // Weights of the layer
    weight_changes: Vec<Vec<f32>>,  // Bias minibatch changes
    biases: Vec<f32>,               // Biases of the layer
    bias_changes: Vec<f32>,         // Bias minibatch changes
    input: Vec<f32>,                // Input vector
    output: Vec<f32>,               // Output vector
}

impl FullyConnectedLayer {
    /// Creates a new fully connected layer with the given input width, input depth, and output
    pub fn new(input_width: usize, input_depth: usize, output_width: usize, output_depth: usize) -> FullyConnectedLayer {
        // Calculate output size
        let output_size: usize = output_width * output_width * output_depth;

        // Calculate the input size from the input width and depth
        let input_size: usize = input_depth * (input_width * input_width);
        // Initialize empty vectors for the biases and weights
        let mut biases: Vec<f32> = vec![];
        let mut weights: Vec<Vec<f32>> = vec![vec![]; input_size];
        // Use He initialisation by using a mean of 0.0 and a standard deviation of sqrt(2/input_neurons)
        let normal = Normal::new(0.0, (2.0/(input_size.pow(2) * input_depth) as f32).sqrt()).unwrap();
        // Initialize the biases and weights with random values drawn from the normal distribution
        for _ in 0..output_size {
            biases.push(0.1);
            for i in 0..input_size {
                weights[i].push(normal.sample(&mut rand::thread_rng()));
            }
        }

        // Create and return a new FullyConnectedLayer with the initialized values
        let layer: FullyConnectedLayer = FullyConnectedLayer {
            input_size,
            input_width,
            input_depth,
            output_width,
            output_depth,
            output_size,
            weights,
            weight_changes: vec![vec![0.0; output_size]; input_size],
            biases,
            bias_changes: vec![0.0; output_size],
            input: vec![],
            output: vec![0.0; output_size],
        };

        layer
    }
}

/// Flattens a 3D vector into a 1D vector.
fn flatten(squares: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    let mut flat_data: Vec<f32> = vec![];

    // Flatten the input by iterating through each square, then each row, and extending the flat_data vector.
    for square in squares {
        for row in square {
            flat_data.extend(row);
        }
    }

    flat_data
}

impl Layer for FullyConnectedLayer {
    /// Calculates the output layer by forward propagating the input layer.
    fn forward_propagate(&mut self, matrix_input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Flatten the input matrix into a 1D vector
        let input: Vec<f32> = flatten(matrix_input);
        // Store the input vector for use in backpropgation.
        self.input = input.clone();
        for j in 0..self.output_size {
            self.output[j] = self.biases[j];
            // Loop through each neuron in the input layer and calculate the weighted sum
            for i in 0..self.input_size {
                self.output[j] += input[i] * self.weights[i][j];
            }
            // Apply the sigmoid activation function to the output value
            // self.output[j] = sigmoid(self.output[j]);
            self.output[j] = self.output[j].max(0.0);
        }
        
        // Rearrange the error vector into a 3D structure for the next layer
        let mut output_3d: Vec<Vec<Vec<f32>>> = vec![vec![vec![]; self.output_width]; self.output_depth];
        for i in 0..self.output_depth {
            for j in 0..self.output_width {
                for k in 0..self.output_width {
                    let index: usize = i * self.output_width.pow(2) + j * self.output_width + k;
                    output_3d[i][j].push(self.output[index]);
                }
            }
        }

        output_3d
    }

    /// Back propagates the error through the fully connected layer
    /// and updates the weights and biases accordingly.
    /// Returns the previous layer's error
    fn back_propagate(&mut self, matrix_error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Flatten the error matrix into a 1D vector
        let error: Vec<f32> = flatten(matrix_error);
        // Initialize a vector to hold the error for each input
        let mut flat_error: Vec<f32> = vec![0.0; self.input_size];

        // Update the biases and weights for each output neuron
        for j in 0..self.output_size {
            if self.output[j] >= 0.0 {
                // Update the bias for the current neuron
                self.bias_changes[j] -= error[j] * LEARNING_RATE;

                // Update the weights for each input neuron connected to the current output neuron
                for i in 0..self.input_size {
                    flat_error[i] += error[j] * self.weights[i][j];
                    // self.weights[i][j] -=
                    //     error[j] * self.input[i] * inv_deriv_sigmoid(self.output[j]) * LEARNING_RATE;
                    self.weight_changes[i][j] -=
                        error[j] * self.input[i] * LEARNING_RATE;
                }
            }
        }

        // Rearrange the error vector into a 3D structure for the previous layer
        let mut prev_error: Vec<Vec<Vec<f32>>> =
            vec![vec![vec![]; self.input_width]; self.input_depth];
        for i in 0..self.input_depth {
            for j in 0..self.input_width {
                for k in 0..self.input_width {
                    let index: usize = i * self.input_width.pow(2) + j * self.input_width + k;
                    prev_error[i][j].push(flat_error[index]);
                }
            }
        }

        prev_error
    }

    /// Returns the output at an index of the fully connected layer.
    fn get_output(&mut self, _index: (usize, usize, usize)) -> f32 {
        panic!("Fully connected layers should not be accessed directly.")
    }

    fn update_layer(&mut self, minibatch_size: usize) {
        for j in 0..self.output_size {
            self.biases[j] += self.bias_changes[j] / minibatch_size as f32;
            self.bias_changes[j] = 0.0;
            for i in 0..self.input_size {
                self.weights[i][j] += self.weight_changes[i][j] / minibatch_size as f32;
                self.weight_changes[i][j] = 0.0;
            }
        }
    }
}