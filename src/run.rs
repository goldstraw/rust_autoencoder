use mnist::{Mnist, MnistBuilder};
use rand::Rng;

use crate::convolutional_autoencoder::Autoencoder;


/// Runs the Autoencoder on the MNIST dataset
pub fn run() {

    // Load the MNIST dataset
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let unfiltered_train_data: Vec<Vec<Vec<Vec<f32>>>> = format_images(trn_img, 50_000);
    let train_data = filter_ones(unfiltered_train_data, trn_lbl);

    let test_data: Vec<Vec<Vec<Vec<f32>>>> = format_images(tst_img, 10_000);
    let test_labels: Vec<u8> = tst_lbl;

    // Creates a new instance of an Autoencoder
    let mut autoencoder: Autoencoder = Autoencoder::new();
    
    autoencoder.add_conv_layer(28, 1, 3, 3, 1, 1, false);
    autoencoder.add_mxpl_layer(28, 3);
    autoencoder.add_conv_layer(14, 3, 5, 3, 1, 1, false);
    autoencoder.add_mxpl_layer(14, 5);
    autoencoder.add_conv_layer(7, 5, 8, 3, 1, 1, false);
    autoencoder.add_fcl_layer(7, 8, 4, 1);
    autoencoder.add_fcl_layer(4, 1, 7, 8);
    autoencoder.add_conv_layer(7, 8, 5, 3, 1, 1, false);
    autoencoder.add_upscaling_layer(7, 5);
    autoencoder.add_conv_layer(14, 5, 3, 3, 1, 1, false);
    autoencoder.add_upscaling_layer(14, 3);
    autoencoder.add_conv_layer(28, 3, 1, 3, 1, 1, true);
    let minibatch_size: usize = 32;

    for i in 0..100000000 {
        let mut rng = rand::thread_rng();
        let index: usize = rng.gen_range(0..train_data.len());
        autoencoder.forward_propagate(train_data[index].clone());
        if i % minibatch_size == minibatch_size-1 {
            autoencoder.update_layers(minibatch_size);
        }
        
        autoencoder.back_propagate(train_data[index].clone());
        
        if i % 500 == 499 {
            assess_performance(&mut autoencoder, test_data.clone(), test_labels.clone());
        }
    }
}

/// Formats the dataset into a 3D vector
fn format_images(data: Vec<u8>, num_images: usize) -> Vec<Vec<Vec<Vec<f32>>>> {
    let img_width: usize = 28;
    let img_height: usize = 28;

    let mut images: Vec<Vec<Vec<Vec<f32>>>> = vec![];
    for image_count in 0..num_images {
        let mut colour_channels: Vec<Vec<Vec<f32>>> = vec![];
        let mut image: Vec<Vec<f32>> = vec![];
        for h in 0..img_height {
            let mut row: Vec<f32> = vec![];
            for w in 0..img_width {
                let i: usize = (image_count * 28 * 28) + (h * 28) + w;
                row.push(data[i] as f32 / 256.0);
            }
            image.push(row);
        }
        colour_channels.push(image);
        images.push(colour_channels);
    }

    images
}

// Filter out all images that are not 1
fn filter_ones(data: Vec<Vec<Vec<Vec<f32>>>>, labels: Vec<u8>) -> Vec<Vec<Vec<Vec<f32>>>> {
    let mut filtered_data: Vec<Vec<Vec<Vec<f32>>>> = vec![];
    for i in 0..data.len() {
        if labels[i] == 1 {
            filtered_data.push(data[i].clone());
        }
    }
    filtered_data
}


// Assess the performance of the Autoencoder on the test data
fn assess_performance(autoencoder: &mut Autoencoder, test_data: Vec<Vec<Vec<Vec<f32>>>>, test_labels: Vec<u8>) {
    let mut total_false_cost: f32 = 0.0;
    let mut total_true_cost: f32 = 0.0;
    let mut false_count = 0;
    let mut true_count = 0;

    for i in 0..test_data.len() {
        autoencoder.forward_propagate(test_data[i].clone());
        if test_labels[i] == 1 {
            total_true_cost += autoencoder.cost(test_data[i].clone());
            true_count += 1;
        } else {
            total_false_cost += autoencoder.cost(test_data[i].clone());
            false_count += 1;
        }
    }

    let avg_false_cost: f32 = total_false_cost / false_count as f32;
    let avg_true_cost: f32 = total_true_cost / true_count as f32;

    println!("Average cost of false images: {}", avg_false_cost);
    println!("Average cost of true images: {}\n", avg_true_cost);
}