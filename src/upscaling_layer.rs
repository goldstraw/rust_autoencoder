use crate::layer::Layer;

/// Defines a `UpscalingLayer` structure.
pub struct UpscalingLayer {
    input_size: usize,
    input_depth: usize,
    output_size: usize,
    output: Vec<Vec<Vec<f32>>>,
}

impl UpscalingLayer {
    /// Create a new max pooling layer with the given parameters
    pub fn new(
        input_size: usize,
        input_depth: usize,
    ) -> UpscalingLayer {
        
        let output_size = input_size * 2;
        // Initialize the output vector with zeros
        let output = vec![vec![vec![0.0; output_size]; output_size]; input_depth];

        // Create a new UpscalingLayer with the initialized parameters and vectors
        let layer: UpscalingLayer = UpscalingLayer {
            input_size,
            input_depth,
            output_size,
            output,
        };

        layer
    }
}

impl Layer for UpscalingLayer {
    /// Expands each input neuron into a 2x2 field of identical values.
    fn forward_propagate(&mut self, input: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Loop through each output position in the output volume
        for y in 0..self.output_size {
            for x in 0..self.output_size {
                let y_i: usize = y / 2;
                let x_i: usize = x / 2;
                for f in 0..self.input_depth {
                    self.output[f][y][x] = input[f][y_i][x_i];
                }
            }
        }
        self.output.clone()
    }

    /// Back propagates the error in an upscaling layer. 
    /// Takes in the error matrix and returns the previous error matrix
    fn back_propagate(&mut self, error: Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // Initialize the previous error vector
        let mut prev_error = vec![vec![vec![0.0; self.input_size]; self.input_size]; self.input_depth];
        // Iterate through the input neurons
        for y in 0..self.input_size {
            for x in 0..self.input_size {
                // Input depth will always be the same as output depth
                for f in 0..self.input_depth {
                    // All errors will be identical, so takes the first neuron's error.
                    prev_error[f][y][x] = error[f][y*2][x*2];
                }
            }
        }

        // Return the previous error vector
        prev_error
    }

    fn get_output(&mut self, _index: (usize, usize, usize)) -> f32 {
        panic!("Upscaling layers should not be accessed directly.")
    }

    fn update_layer(&mut self, _minibatch_size: usize) {
        ()
    }
}