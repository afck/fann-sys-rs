//! Raw bindings to C functions of the Fast Artificial Neural Network library
#![allow(non_camel_case_types)]

use libc::types::common::c95::FILE;
use libc::{c_char, c_float, c_int, c_uint, c_void};

pub type fann_type = c_float;

/// Error events on fann and fann_train_data.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_errno_enum {
    /// No error
	FANN_E_NO_ERROR = 0,
    /// Unable to open configuration file for reading
	FANN_E_CANT_OPEN_CONFIG_R,
	/// Unable to open configuration file for writing
	FANN_E_CANT_OPEN_CONFIG_W,
    /// Wrong version of configuration file
	FANN_E_WRONG_CONFIG_VERSION,
	/// Error reading info from configuration file
	FANN_E_CANT_READ_CONFIG,
	/// Error reading neuron info from configuration file
	FANN_E_CANT_READ_NEURON,
	/// Error reading connections from configuration file
	FANN_E_CANT_READ_CONNECTIONS,
	/// Number of connections not equal to the number expected
	FANN_E_WRONG_NUM_CONNECTIONS,
	/// Unable to open train data file for writing
	FANN_E_CANT_OPEN_TD_W,
	/// Unable to open train data file for reading
	FANN_E_CANT_OPEN_TD_R,
	/// Error reading training data from file
	FANN_E_CANT_READ_TD,
	/// Unable to allocate memory
	FANN_E_CANT_ALLOCATE_MEM,
	/// Unable to train with the selected activation function
	FANN_E_CANT_TRAIN_ACTIVATION,
	/// Unable to use the selected activation function
	FANN_E_CANT_USE_ACTIVATION,
	/// Irreconcilable differences between two fann_train_data structures
	FANN_E_TRAIN_DATA_MISMATCH,
	/// Unable to use the selected training algorithm
	FANN_E_CANT_USE_TRAIN_ALG,
	/// Trying to take subset which is not within the training set
	FANN_E_TRAIN_DATA_SUBSET,
	/// Index is out of bound
	FANN_E_INDEX_OUT_OF_BOUND,
	/// Scaling parameters not present
	FANN_E_SCALE_NOT_PRESENT,
}

/// The Training algorithms used when training on fann_train_data with functions like
/// fann_train_on_data or fann_train_on_file. The incremental training looks alters the weights
/// after each time it is presented an input pattern, while batch only alters the weights once after
/// it has been presented to all the patterns.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_train_enum {
	/// Standard backpropagation algorithm, where the weights are
    /// updated after each training pattern. This means that the weights are updated many
    /// times during a single epoch. For this reason some problems, will train very fast with
    /// this algorithm, while other more advanced problems will not train very well.
	FANN_TRAIN_INCREMENTAL = 0,
	/// Standard backpropagation algorithm, where the weights are updated after calculating the mean
    /// square error for the whole training set. This means that the weights are only updated once
    /// during a epoch. For this reason some problems, will train slower with this algorithm. But
    /// since the mean square error is calculated more correctly than in incremental training, some
    /// problems will reach a better solutions with this algorithm.
	FANN_TRAIN_BATCH,
	/// A more advanced batch training algorithm which achieves good results
    /// for many problems. The RPROP training algorithm is adaptive, and does therefore not
    /// use the learning_rate. Some other parameters can however be set to change the way the
    /// RPROP algorithm works, but it is only recommended for users with insight in how the RPROP
    /// training algorithm works. The RPROP training algorithm is described by
    /// [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the
    /// iRPROP- training algorithm which is described by [Igel and Husken, 2000] which
    /// is an variety of the standard RPROP training algorithm.
    FANN_TRAIN_RPROP,
    /// A more advanced batch training algorithm which achieves good results
    /// for many problems. The quickprop training algorithm uses the learning_rate parameter
    /// along with other more advanced parameters, but it is only recommended to change these
    /// advanced parameters, for users with insight in how the quickprop training algorithm works.
    /// The quickprop training algorithm is described by [Fahlman, 1988].
	FANN_TRAIN_QUICKPROP,
}

/// The activation functions used for the neurons during training. The activation functions
/// can either be defined for a group of neurons by fann_set_activation_function_hidden and
/// fann_set_activation_function_output or it can be defined for a single neuron by
/// fann_set_activation_function.
///
/// The steepness of an activation function is defined in the same way by
/// fann_set_activation_steepness_hidden, fann_set_activation_steepness_output and
/// fann_set_activation_steepness.
///
/// The functions are described with functions where:
///
/// * x is the input to the activation function,
///
/// * y is the output,
///
/// * s is the steepness and
///
/// * d is the derivation.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_activationfunc_enum {
    /// Linear activation function.
    ///
    /// * span: -inf < y < inf
    ///
    /// * y = x*s, d = 1*s
    ///
    /// * Can NOT be used in fixed point.
	FANN_LINEAR = 0,
    /// Threshold activation function.
    ///
    /// * x < 0 -> y = 0, x >= 0 -> y = 1
    ///
    /// * Can NOT be used during training.
	FANN_THRESHOLD,
    /// Threshold activation function.
    ///
    /// * x < 0 -> y = 0, x >= 0 -> y = 1
    ///
    /// * Can NOT be used during training.
	FANN_THRESHOLD_SYMMETRIC,
    /// Sigmoid activation function.
    ///
    /// * One of the most used activation functions.
    ///
    /// * span: 0 < y < 1
    ///
    /// * y = 1/(1 + exp(-2*s*x))
    ///
    /// * d = 2*s*y*(1 - y)
	FANN_SIGMOID,
    /// Stepwise linear approximation to sigmoid.
    ///
    /// * Faster than sigmoid but a bit less precise.
	FANN_SIGMOID_STEPWISE,
    /// Symmetric sigmoid activation function, aka. tanh.
    ///
    /// * One of the most used activation functions.
    ///
    /// * span: -1 < y < 1
    ///
    /// * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
    ///
    /// * d = s*(1-(y*y))
	FANN_SIGMOID_SYMMETRIC,
    /// Stepwise linear approximation to symmetric sigmoid.
    ///
    /// * Faster than symmetric sigmoid but a bit less precise.
	FANN_SIGMOID_SYMMETRIC_STEPWISE,
    /// Gaussian activation function.
    ///
    /// * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
    ///
    /// * span: 0 < y < 1
    ///
    /// * y = exp(-x*s*x*s)
    ///
    /// * d = -2*x*s*y*s
	FANN_GAUSSIAN,
    /// Symmetric gaussian activation function.
    ///
    /// * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
    ///
    /// * span: -1 < y < 1
    ///
    /// * y = exp(-x*s*x*s)*2-1
    ///
    /// * d = -2*x*s*(y+1)*s
	FANN_GAUSSIAN_SYMMETRIC,
    /// Stepwise linear approximation to gaussian.
    /// Faster than gaussian but a bit less precise.
    /// NOT implemented yet.
	FANN_GAUSSIAN_STEPWISE,
    /// Fast (sigmoid like) activation function defined by David Elliott
    ///
    /// * span: 0 < y < 1
    ///
    /// * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
    ///
    /// * d = s*1/(2*(1+|x*s|)*(1+|x*s|))
	FANN_ELLIOT,
    /// Fast (symmetric sigmoid like) activation function defined by David Elliott
    ///
    /// * span: -1 < y < 1
    ///
    /// * y = (x*s) / (1 + |x*s|)
    ///
    /// * d = s*1/((1+|x*s|)*(1+|x*s|))
	FANN_ELLIOT_SYMMETRIC,
    /// Bounded linear activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = x*s, d = 1*s
	FANN_LINEAR_PIECE,
    /// Bounded linear activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = x*s, d = 1*s
	FANN_LINEAR_PIECE_SYMMETRIC,
    /// Periodical sinus activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = sin(x*s)
    ///
    /// * d = s*cos(x*s)
	FANN_SIN_SYMMETRIC,
    /// Periodical cosinus activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = cos(x*s)
    ///
    /// * d = s*-sin(x*s)
	FANN_COS_SYMMETRIC,
    /// Periodical sinus activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = sin(x*s)/2+0.5
    ///
    /// * d = s*cos(x*s)/2
	FANN_SIN,
    /// Periodical cosinus activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = cos(x*s)/2+0.5
    ///
    /// * d = s*-sin(x*s)/2
	FANN_COS,
}

///	Error function used during training.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_errorfunc_enum {
    /// Standard linear error function.
	FANN_ERRORFUNC_LINEAR = 0,
    /// Tanh error function, usually better but can require a lower learning rate. This error
    /// function agressively targets outputs that differ much from the desired, while not targetting
    /// outputs that only differ a little that much. This activation function is not recommended for
    /// cascade training and incremental training.
	FANN_ERRORFUNC_TANH,
}

/// Stop criteria used during training.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_stopfunc_enum {
	/// Stop criteria is Mean Square Error (MSE) value.
	FANN_STOPFUNC_MSE = 0,
	/// Stop criteria is number of bits that fail. The number of bits; means the
    /// number of output neurons which differ more than the bit fail limit
    /// (see fann_get_bit_fail_limit, fann_set_bit_fail_limit).
    /// The bits are counted in all of the training data, so this number can be higher than
    /// the number of training data.
	FANN_STOPFUNC_BIT,
}

/// Definition of network types used by <fann_get_network_type>
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_nettype_enum {
    /// Each layer only has connections to the next layer
    FANN_NETTYPE_LAYER = 0,
    /// Each layer has connections to all following layers
    FANN_NETTYPE_SHORTCUT,
}

/// This callback function can be called during training when using fann_train_on_data,
/// fann_train_on_file or fann_cascadetrain_on_data.
///
/// The callback can be set by using fann_set_callback and is very useful for doing custom
/// things during training. It is recommended to use this function when implementing custom
/// training procedures, or when visualizing the training in a GUI etc. The parameters which the
/// callback function takes is the parameters given to the fann_train_on_data, plus an epochs
/// parameter which tells how many epochs the training have taken so far.
///
/// The callback function should return an integer, if the callback function returns -1, the
/// training will terminate.
// TODO: Translate the example to rust.
//
// Example of a callback function (in C):
//
// ```c
// int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
//                            unsigned int max_epochs, unsigned int epochs_between_reports,
//                            float desired_error, unsigned int epochs)
// {
// 	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
// 	return 0;
// }
// ```
pub type fann_callback_type = Option<
    extern "C" fn(ann: *mut fann,
                  train: *mut fann_train_data,
                  max_epochs: c_uint,
                  epochs_between_reports: c_uint,
                  desired_error: c_float,
                  epochs: c_uint) -> c_int>;

#[repr(C)]
#[derive(Copy)]
pub struct fann_neuron {
    first_con: c_uint,
    last_con: c_uint,
    sum: fann_type,
    value: fann_type,
    activation_steepness: fann_type,
    activation_function: fann_activationfunc_enum,
}

impl ::std::clone::Clone for fann_neuron {
    fn clone(&self) -> Self { *self }
}

#[repr(C)]
#[derive(Copy)]
pub struct fann_layer {
    first_neuron: *mut fann_neuron,
    last_neuron: *mut fann_neuron,
}

impl ::std::clone::Clone for fann_layer {
    fn clone(&self) -> Self { *self }
}

#[repr(C)]
#[derive(Copy)]
pub struct fann_error {
    errno_f: fann_errno_enum,
    error_log: *mut FILE,
    errstr: *mut c_char,
}

impl ::std::clone::Clone for fann_error {
    fn clone(&self) -> Self { *self }
}

#[repr(C)]
#[derive(Copy)]
pub struct fann {
    errno_f: fann_errno_enum,
    error_log: *mut FILE,
    errstr: *mut c_char,
    learning_rate: c_float,
    learning_momentum: c_float,
    connection_rate: c_float,
    network_type: fann_nettype_enum,
    first_layer: *mut fann_layer,
    last_layer: *mut fann_layer,
    total_neurons: c_uint,
    num_input: c_uint,
    num_output: c_uint,
    weights: *mut fann_type,
    connections: *mut *mut fann_neuron,
    train_errors: *mut fann_type,
    training_algorithm: fann_train_enum,
    total_connections: c_uint,
    output: *mut fann_type,
    num_mse: c_uint,
    mse_value: c_float,
    num_bit_fail: c_uint,
    bit_fail_limit: fann_type,
    train_error_function: fann_errorfunc_enum,
    train_stop_function: fann_stopfunc_enum,
    callback: fann_callback_type,
    user_data: *mut c_void,
    cascade_output_change_fraction: c_float,
    cascade_output_stagnation_epochs: c_uint,
    cascade_candidate_change_fraction: c_float,
    cascade_candidate_stagnation_epochs: c_uint,
    cascade_best_candidate: c_uint,
    cascade_candidate_limit: fann_type,
    cascade_weight_multiplier: fann_type,
    cascade_max_out_epochs: c_uint,
    cascade_max_cand_epochs: c_uint,
    cascade_activation_functions: *mut fann_activationfunc_enum,
    cascade_activation_functions_count: c_uint,
    cascade_activation_steepnesses: *mut fann_type,
    cascade_activation_steepnesses_count: c_uint,
    cascade_num_candidate_groups: c_uint,
    cascade_candidate_scores: *mut fann_type,
    total_neurons_allocated: c_uint,
    total_connections_allocated: c_uint,
    quickprop_decay: c_float,
    quickprop_mu: c_float,
    rprop_increase_factor: c_float,
    rprop_decrease_factor: c_float,
    rprop_delta_min: c_float,
    rprop_delta_max: c_float,
    rprop_delta_zero: c_float,
    train_slopes: *mut fann_type,
    prev_steps: *mut fann_type,
    prev_train_slopes: *mut fann_type,
    prev_weights_deltas: *mut fann_type,
    scale_mean_in: *mut c_float,
    scale_deviation_in: *mut c_float,
    scale_new_min_in: *mut c_float,
    scale_factor_in: *mut c_float,
    scale_mean_out: *mut c_float,
    scale_deviation_out: *mut c_float,
    scale_new_min_out: *mut c_float,
    scale_factor_out: *mut c_float,
}

impl ::std::clone::Clone for fann {
    fn clone(&self) -> Self { *self }
}

#[repr(C)]
#[derive(Copy)]
pub struct fann_connection {
    from_neuron: c_uint,
    to_neuron: c_uint,
    weight: fann_type,
}

impl ::std::clone::Clone for fann_connection {
    fn clone(&self) -> Self { *self }
}

#[repr(C)]
#[derive(Copy)]
pub struct fann_train_data {
    errno_f: fann_errno_enum,
    error_log: *mut FILE,
    errstr: *mut c_char,
    num_data: c_uint,
    num_input: c_uint,
    num_output: c_uint,
    input: *mut *mut fann_type,
    output: *mut *mut fann_type,
}

impl ::std::clone::Clone for fann_train_data {
    fn clone(&self) -> Self { *self }
}

// TODO: Copy documentation for the remaining functions.
#[link(name = "fann")]
extern "C" {
    pub static mut fann_default_error_log: *mut FILE;

    pub fn fann_set_error_log(errdat: *mut fann_error, log_file: *mut FILE);
    pub fn fann_get_errno(errdat: *mut fann_error) -> fann_errno_enum;
    pub fn fann_reset_errno(errdat: *mut fann_error);
    pub fn fann_reset_errstr(errdat: *mut fann_error);
    pub fn fann_get_errstr(errdat: *mut fann_error) -> *mut c_char;
    pub fn fann_print_error(errdat: *mut fann_error);
    pub fn fann_allocate_structure(num_layers: c_uint) -> *mut fann;
    pub fn fann_allocate_neurons(ann: *mut fann);
    pub fn fann_allocate_connections(ann: *mut fann);
    pub fn fann_save_internal(ann: *mut fann,
                              configuration_file: *const c_char,
                              save_as_fixed: c_uint) -> c_int;
    pub fn fann_save_internal_fd(ann: *mut fann, conf: *mut FILE,
                                 configuration_file: *const c_char,
                                 save_as_fixed: c_uint) -> c_int;
    pub fn fann_save_train_internal(data: *mut fann_train_data,
                                    filename: *const c_char,
                                    save_as_fixed: c_uint,
                                    decimal_point: c_uint) -> c_int;
    pub fn fann_save_train_internal_fd(data: *mut fann_train_data,
                                       file: *mut FILE,
                                       filename: *const c_char,
                                       save_as_fixed: c_uint,
                                       decimal_point: c_uint) -> c_int;
    pub fn fann_update_stepwise(ann: *mut fann);
    pub fn fann_seed_rand();
    pub fn fann_error(errdat: *mut fann_error, errno_f: fann_errno_enum, ...);
    pub fn fann_init_error_data(errdat: *mut fann_error);
    pub fn fann_create_from_fd(conf: *mut FILE, configuration_file: *const c_char)
        -> *mut fann;
    pub fn fann_read_train_from_fd(file: *mut FILE, filename: *const c_char)
        -> *mut fann_train_data;
    pub fn fann_compute_MSE(ann: *mut fann, desired_output: *mut fann_type);
    pub fn fann_update_output_weights(ann: *mut fann);
    pub fn fann_backpropagate_MSE(ann: *mut fann);
    pub fn fann_update_weights(ann: *mut fann);
    pub fn fann_update_slopes_batch(ann: *mut fann,
                                    layer_begin: *mut fann_layer,
                                    layer_end: *mut fann_layer);
    pub fn fann_update_weights_quickprop(ann: *mut fann,
                                         num_data: c_uint,
                                         first_weight: c_uint,
                                         past_end: c_uint);
    pub fn fann_update_weights_batch(ann: *mut fann,
                                     num_data: c_uint,
                                     first_weight: c_uint,
                                     past_end: c_uint);
    pub fn fann_update_weights_irpropm(ann: *mut fann,
                                       first_weight: c_uint,
                                       past_end: c_uint);
    pub fn fann_clear_train_arrays(ann: *mut fann);
    pub fn fann_activation(ann: *mut fann,
                           activation_function: c_uint,
                           steepness: fann_type, value: fann_type) -> fann_type;
    pub fn fann_activation_derived(activation_function: c_uint,
                                   steepness: fann_type, value: fann_type,
                                   sum: fann_type) -> fann_type;
    pub fn fann_desired_error_reached(ann: *mut fann,
                                      desired_error: c_float) -> c_int;
    pub fn fann_train_outputs(ann: *mut fann,
                              data: *mut fann_train_data,
                              desired_error: c_float) -> c_int;
    pub fn fann_train_outputs_epoch(ann: *mut fann,
                                    data: *mut fann_train_data) -> c_float;
    pub fn fann_train_candidates(ann: *mut fann,
                                 data: *mut fann_train_data) -> c_int;
    pub fn fann_train_candidates_epoch(ann: *mut fann,
                                       data: *mut fann_train_data) -> fann_type;
    pub fn fann_install_candidate(ann: *mut fann);
    pub fn fann_initialize_candidates(ann: *mut fann) -> c_int;
    pub fn fann_set_shortcut_connections(ann: *mut fann);
    pub fn fann_allocate_scale(ann: *mut fann) -> c_int;
    pub fn fann_train(ann: *mut fann, input: *mut fann_type,
                      desired_output: *mut fann_type);
    pub fn fann_test(ann: *mut fann, input: *mut fann_type,
                     desired_output: *mut fann_type) -> *mut fann_type;
    pub fn fann_get_MSE(ann: *mut fann) -> c_float;
    pub fn fann_get_bit_fail(ann: *mut fann) -> c_uint;
    pub fn fann_reset_MSE(ann: *mut fann);
    pub fn fann_train_on_data(ann: *mut fann,
                              data: *mut fann_train_data,
                              max_epochs: c_uint,
                              epochs_between_reports: c_uint,
                              desired_error: c_float);
    pub fn fann_train_on_file(ann: *mut fann,
                              filename: *const c_char,
                              max_epochs: c_uint,
                              epochs_between_reports: c_uint,
                              desired_error: c_float);
    pub fn fann_train_epoch(ann: *mut fann, data: *mut fann_train_data) -> c_float;
    pub fn fann_test_data(ann: *mut fann, data: *mut fann_train_data) -> c_float;
    pub fn fann_read_train_from_file(filename: *const c_char) -> *mut fann_train_data;
    pub fn fann_create_train_from_callback(num_data: c_uint, num_input: c_uint, num_output: c_uint,
        user_function: Option<extern "C" fn(arg1: c_uint, arg2: c_uint, arg3: c_uint,
                                            arg4: *mut fann_type, arg5: *mut fann_type)>)
        -> *mut fann_train_data;
    pub fn fann_destroy_train(train_data: *mut fann_train_data);
    pub fn fann_shuffle_train_data(train_data: *mut fann_train_data);
    pub fn fann_scale_train(ann: *mut fann, data: *mut fann_train_data);
    pub fn fann_descale_train(ann: *mut fann, data: *mut fann_train_data);
    pub fn fann_set_input_scaling_params(ann: *mut fann,
                                         data: *const fann_train_data,
                                         new_input_min: c_float,
                                         new_input_max: c_float) -> c_int;
    pub fn fann_set_output_scaling_params(ann: *mut fann,
                                          data: *const fann_train_data,
                                          new_output_min: c_float,
                                          new_output_max: c_float) -> c_int;
    pub fn fann_set_scaling_params(ann: *mut fann,
                                   data: *const fann_train_data,
                                   new_input_min: c_float,
                                   new_input_max: c_float,
                                   new_output_min: c_float,
                                   new_output_max: c_float) -> c_int;
    pub fn fann_clear_scaling_params(ann: *mut fann) -> c_int;
    pub fn fann_scale_input(ann: *mut fann, input_vector: *mut fann_type);
    pub fn fann_scale_output(ann: *mut fann, output_vector: *mut fann_type);
    pub fn fann_descale_input(ann: *mut fann, input_vector: *mut fann_type);
    pub fn fann_descale_output(ann: *mut fann, output_vector: *mut fann_type);
    pub fn fann_scale_input_train_data(train_data: *mut fann_train_data,
                                       new_min: fann_type, new_max: fann_type);
    pub fn fann_scale_output_train_data(train_data: *mut fann_train_data,
                                        new_min: fann_type, new_max: fann_type);
    pub fn fann_scale_train_data(train_data: *mut fann_train_data,
                                 new_min: fann_type, new_max: fann_type);
    pub fn fann_merge_train_data(data1: *mut fann_train_data,
                                 data2: *mut fann_train_data) -> *mut fann_train_data;
    pub fn fann_duplicate_train_data(data: *mut fann_train_data)
        -> *mut fann_train_data;
    pub fn fann_subset_train_data(data: *mut fann_train_data, pos: c_uint, length: c_uint)
        -> *mut fann_train_data;
    pub fn fann_length_train_data(data: *mut fann_train_data) -> c_uint;
    pub fn fann_num_input_train_data(data: *mut fann_train_data) -> c_uint;
    pub fn fann_num_output_train_data(data: *mut fann_train_data) -> c_uint;
    pub fn fann_save_train(data: *mut fann_train_data, filename: *const c_char) -> c_int;
    pub fn fann_save_train_to_fixed(data: *mut fann_train_data,
                                    filename: *const c_char,
                                    decimal_point: c_uint) -> c_int;
    pub fn fann_get_training_algorithm(ann: *mut fann) -> fann_train_enum;
    pub fn fann_set_training_algorithm(ann: *mut fann,
                                       training_algorithm: fann_train_enum);
    pub fn fann_get_learning_rate(ann: *mut fann) -> c_float;
    pub fn fann_set_learning_rate(ann: *mut fann, learning_rate: c_float);
    pub fn fann_get_learning_momentum(ann: *mut fann) -> c_float;
    pub fn fann_set_learning_momentum(ann: *mut fann, learning_momentum: c_float);
    pub fn fann_get_activation_function(ann: *mut fann, layer: c_int, neuron: c_int)
        -> fann_activationfunc_enum;
    pub fn fann_set_activation_function(ann: *mut fann,
                                        activation_function: fann_activationfunc_enum,
                                        layer: c_int,
                                        neuron: c_int);
    pub fn fann_set_activation_function_layer(ann: *mut fann,
                                              activation_function: fann_activationfunc_enum,
                                              layer: c_int);
    pub fn fann_set_activation_function_hidden(ann: *mut fann,
                                               activation_function: fann_activationfunc_enum);
    pub fn fann_set_activation_function_output(ann: *mut fann,
                                               activation_function: fann_activationfunc_enum);
    pub fn fann_get_activation_steepness(ann: *mut fann, layer: c_int, neuron: c_int)
        -> fann_type;
    pub fn fann_set_activation_steepness(ann: *mut fann,
                                         steepness: fann_type,
                                         layer: c_int,
                                         neuron: c_int);
    pub fn fann_set_activation_steepness_layer(ann: *mut fann,
                                               steepness: fann_type,
                                               layer: c_int);
    pub fn fann_set_activation_steepness_hidden(ann: *mut fann,
                                                steepness: fann_type);
    pub fn fann_set_activation_steepness_output(ann: *mut fann,
                                                steepness: fann_type);
    pub fn fann_get_train_error_function(ann: *mut fann) -> fann_errorfunc_enum;
    pub fn fann_set_train_error_function(ann: *mut fann,
                                         train_error_function: fann_errorfunc_enum);
    pub fn fann_get_train_stop_function(ann: *mut fann) -> fann_stopfunc_enum;
    pub fn fann_set_train_stop_function(ann: *mut fann,
                                        train_stop_function: fann_stopfunc_enum);
    pub fn fann_get_bit_fail_limit(ann: *mut fann) -> fann_type;
    pub fn fann_set_bit_fail_limit(ann: *mut fann, bit_fail_limit: fann_type);
    pub fn fann_set_callback(ann: *mut fann, callback: fann_callback_type);
    pub fn fann_get_quickprop_decay(ann: *mut fann) -> c_float;
    pub fn fann_set_quickprop_decay(ann: *mut fann, quickprop_decay: c_float);
    pub fn fann_get_quickprop_mu(ann: *mut fann) -> c_float;
    pub fn fann_set_quickprop_mu(ann: *mut fann, quickprop_mu: c_float);
    pub fn fann_get_rprop_increase_factor(ann: *mut fann) -> c_float;
    pub fn fann_set_rprop_increase_factor(ann: *mut fann, rprop_increase_factor: c_float);
    pub fn fann_get_rprop_decrease_factor(ann: *mut fann) -> c_float;
    pub fn fann_set_rprop_decrease_factor(ann: *mut fann, rprop_decrease_factor: c_float);
    pub fn fann_get_rprop_delta_min(ann: *mut fann) -> c_float;
    pub fn fann_set_rprop_delta_min(ann: *mut fann, rprop_delta_min: c_float);
    pub fn fann_get_rprop_delta_max(ann: *mut fann) -> c_float;
    pub fn fann_set_rprop_delta_max(ann: *mut fann, rprop_delta_max: c_float);
    pub fn fann_get_rprop_delta_zero(ann: *mut fann) -> c_float;
    pub fn fann_set_rprop_delta_zero(ann: *mut fann, rprop_delta_max: c_float);
    pub fn fann_cascadetrain_on_data(ann: *mut fann,
                                     data: *mut fann_train_data,
                                     max_neurons: c_uint,
                                     neurons_between_reports: c_uint,
                                     desired_error: c_float);
    pub fn fann_cascadetrain_on_file(ann: *mut fann,
                                     filename: *const c_char,
                                     max_neurons: c_uint,
                                     neurons_between_reports: c_uint,
                                     desired_error: c_float);
    pub fn fann_get_cascade_output_change_fraction(ann: *mut fann) -> c_float;
    pub fn fann_set_cascade_output_change_fraction(ann: *mut fann,
                                                   cascade_output_change_fraction: c_float);
    pub fn fann_get_cascade_output_stagnation_epochs(ann: *mut fann) -> c_uint;
    pub fn fann_set_cascade_output_stagnation_epochs(ann: *mut fann,
                                                     cascade_output_stagnation_epochs: c_uint)
    ;
    pub fn fann_get_cascade_candidate_change_fraction(ann: *mut fann) -> c_float;
    pub fn fann_set_cascade_candidate_change_fraction(ann: *mut fann,
                                                      cascade_candidate_change_fraction: c_float)
    ;
    pub fn fann_get_cascade_candidate_stagnation_epochs(ann: *mut fann) -> c_uint;
    pub fn fann_set_cascade_candidate_stagnation_epochs(ann: *mut fann,
        cascade_candidate_stagnation_epochs: c_uint);
    pub fn fann_get_cascade_weight_multiplier(ann: *mut fann) -> fann_type;
    pub fn fann_set_cascade_weight_multiplier(ann: *mut fann,
                                              cascade_weight_multiplier: fann_type);
    pub fn fann_get_cascade_candidate_limit(ann: *mut fann) -> fann_type;
    pub fn fann_set_cascade_candidate_limit(ann: *mut fann,
                                            cascade_candidate_limit: fann_type);
    pub fn fann_get_cascade_max_out_epochs(ann: *mut fann) -> c_uint;
    pub fn fann_set_cascade_max_out_epochs(ann: *mut fann,
                                           cascade_max_out_epochs: c_uint);
    pub fn fann_get_cascade_max_cand_epochs(ann: *mut fann) -> c_uint;
    pub fn fann_set_cascade_max_cand_epochs(ann: *mut fann,
                                            cascade_max_cand_epochs: c_uint);
    pub fn fann_get_cascade_num_candidates(ann: *mut fann) -> c_uint;
    pub fn fann_get_cascade_activation_functions_count(ann: *mut fann) -> c_uint;
    pub fn fann_get_cascade_activation_functions(ann: *mut fann)
        -> *mut fann_activationfunc_enum;
    pub fn fann_set_cascade_activation_functions(ann: *mut fann,
        cascade_activation_functions: *mut fann_activationfunc_enum,                                                 cascade_activation_functions_count: c_uint);
    pub fn fann_get_cascade_activation_steepnesses_count(ann: *mut fann) -> c_uint;
    pub fn fann_get_cascade_activation_steepnesses(ann: *mut fann) -> *mut fann_type;
    pub fn fann_set_cascade_activation_steepnesses(ann: *mut fann,
        cascade_activation_steepnesses: *mut fann_type,
        cascade_activation_steepnesses_count: c_uint);
    pub fn fann_get_cascade_num_candidate_groups(ann: *mut fann) -> c_uint;
    pub fn fann_set_cascade_num_candidate_groups(ann: *mut fann,
                                                 cascade_num_candidate_groups: c_uint);
    pub fn fann_create_from_file(configuration_file: *const c_char) -> *mut fann;
    pub fn fann_save(ann: *mut fann, configuration_file: *const c_char) -> c_int;
    pub fn fann_save_to_fixed(ann: *mut fann, configuration_file: *const c_char) -> c_int;

    /// Creates a standard fully connected backpropagation neural network.
    ///
    /// There will be a bias neuron in each layer (except the output layer),
    /// and this bias neuron will be connected to all neurons in the next layer.
    /// When running the network, the bias nodes always emits 1.
    ///
    /// To destroy a fann use the fann_destroy function.
    ///
    /// This function appears in FANN >= 2.0.0.
    ///
    /// # Parameters:
    ///
    /// * `num_layers` - The total number of layers including the input and the output layer.
    /// * `...` - Integer values determining the number of neurons in each layer starting with the
    /// input layer and ending with the output layer.
    ///
    /// # Returns:
    ///
    /// A pointer to the newly created fann.
    ///
    /// # Example:
    ///
    /// Creating an ANN with 2 input neurons, 1 output neuron,
    /// and two hidden layers with 8 and 9 neurons
    ///
    /// ```
    /// unsafe {
    ///     let ann = fann::ffi::fann_create_standard(4, 2, 8, 9, 1);
    /// }
    /// ```
    pub fn fann_create_standard(num_layers: c_uint, ...) -> *mut fann;

    pub fn fann_create_standard_array(num_layers: c_uint, layers: *const c_uint)
        -> *mut fann;
    pub fn fann_create_sparse(connection_rate: c_float, num_layers: c_uint, ...)
        -> *mut fann;
    pub fn fann_create_sparse_array(connection_rate: c_float,
                                    num_layers: c_uint,
                                    layers: *const c_uint) -> *mut fann;
    pub fn fann_create_shortcut(num_layers: c_uint, ...) -> *mut fann;
    pub fn fann_create_shortcut_array(num_layers: c_uint, layers: *const c_uint)
        -> *mut fann;

    /// Destroys the entire network and properly freeing all the associated memory.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_destroy(ann: *mut fann);

    pub fn fann_run(ann: *mut fann, input: *mut fann_type) -> *mut fann_type;
    pub fn fann_randomize_weights(ann: *mut fann,
                                  min_weight: fann_type,
                                  max_weight: fann_type);
    pub fn fann_init_weights(ann: *mut fann,
                             train_data: *mut fann_train_data);
    pub fn fann_print_connections(ann: *mut fann);
    pub fn fann_print_parameters(ann: *mut fann);
    pub fn fann_get_num_input(ann: *mut fann) -> c_uint;
    pub fn fann_get_num_output(ann: *mut fann) -> c_uint;
    pub fn fann_get_total_neurons(ann: *mut fann) -> c_uint;
    pub fn fann_get_total_connections(ann: *mut fann) -> c_uint;
    pub fn fann_get_network_type(ann: *mut fann) -> fann_nettype_enum;
    pub fn fann_get_connection_rate(ann: *mut fann) -> c_float;
    pub fn fann_get_num_layers(ann: *mut fann) -> c_uint;
    pub fn fann_get_layer_array(ann: *mut fann, layers: *mut c_uint);
    pub fn fann_get_bias_array(ann: *mut fann, bias: *mut c_uint);
    pub fn fann_get_connection_array(ann: *mut fann,
                                     connections: *mut fann_connection);
    pub fn fann_set_weight_array(ann: *mut fann,
                                 connections: *mut fann_connection,
                                 num_connections: c_uint);
    pub fn fann_set_weight(ann: *mut fann, from_neuron: c_uint,
                           to_neuron: c_uint, weight: fann_type);
    pub fn fann_set_user_data(ann: *mut fann, user_data: *mut c_void);
    pub fn fann_get_user_data(ann: *mut fann) -> *mut c_void;
}
