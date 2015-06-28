//! Raw bindings to C functions of the Fast Artificial Neural Network library
//!
//!
//! # Creation/Execution
//!
//! The FANN library is designed to be very easy to use.
//! A feedforward ANN can be created by a simple `fann_create_standard` function, while
//! other ANNs can be created just as easily. The ANNs can be trained by `fann_train_on_file`
//! and executed by `fann_run`.
//!
//! All of this can be done without much knowledge of the internals of ANNs, although the ANNs
//! created will still be powerful and effective. If you have more knowledge about ANNs, and desire
//! more control, almost every part of the ANNs can be parametrized to create specialized and highly
//! optimal ANNs.
//!
//!
//! # Training
//!
//! There are many different ways of training neural networks and the FANN library supports
//! a number of different approaches.
//!
//! Two fundementally different approaches are the most commonly used:
//!
//! * Fixed topology training - The size and topology of the ANN is determined in advance
//! and the training alters the weights in order to minimize the difference between
//! the desired output values and the actual output values. This kind of training is
//! supported by `fann_train_on_data`.
//!
//! * Evolving topology training - The training start out with an empty ANN, only consisting
//! of input and output neurons. Hidden neurons and connections are added during training,
//! in order to achieve the same goal as for fixed topology training. This kind of training
//! is supported by FANN Cascade Training.
//!
//!
//! # Cascade Training
//!
//! Cascade training differs from ordinary training in the sense that it starts with an empty neural
//! network and then adds neurons one by one, while it trains the neural network. The main benefit
//! of this approach is that you do not have to guess the number of hidden layers and neurons prior
//! to training, but cascade training has also proved better at solving some problems.
//!
//! The basic idea of cascade training is that a number of candidate neurons are trained separate
//! from the real network, then the most promising of these candidate neurons is inserted into the
//! neural network. Then the output connections are trained and new candidate neurons are prepared.
//! The candidate neurons are created as shortcut connected neurons in a new hidden layer, which
//! means that the final neural network will consist of a number of hidden layers with one shortcut
//! connected neuron in each.
//!
//!
//! # File Input/Output
//!
//! It is possible to save an entire ann to a file with `fann_save` for future loading with
//! `fann_create_from_file`.
//!
//!
//! # Error Handling
//!
//! Errors from the FANN library are usually reported on `stderr`.
//! It is however possible to redirect these error messages to a file,
//! or completely ignore them with the `fann_set_error_log` function.
//!
//! It is also possible to inspect the last error message by using the
//! `fann_get_errno` and `fann_get_errstr` functions.
//!
//!
//! # Datatypes
//!
//! The two main datatypes used in the FANN library are `fann`,
//! which represents an artificial neural network, and `fann_train_data`,
//! which represents training data.
#![allow(non_camel_case_types)]

// TODO: Cross-link the documentation.

extern crate libc;

pub use fann_errno_enum::*;
pub use fann_train_enum::*;
pub use fann_activationfunc_enum::*;
pub use fann_errorfunc_enum::*;
pub use fann_stopfunc_enum::*;
pub use fann_nettype_enum::*;

use libc::types::common::c95::FILE;
use libc::{c_char, c_float, c_int, c_uint, c_void};

/// `fann_type` is the type used for the weights, inputs and outputs of the neural network. In
/// the Rust bindings, it is currently always defined as `c_float`.
///
/// In the FANN C library, `fann_type` is defined as a:
///
/// * `float` - if you include fann.h or floatfann.h
/// * `double` - if you include doublefann.h
/// * `int` - if you include fixedfann.h (please be aware that fixed point usage is
///           only to be used during execution, and not during training).
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
    /// Irreconcilable differences between two `fann_train_data` structures
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

/// The Training algorithms used when training on `fann_train_data` with functions like
/// `fann_train_on_data` or `fann_train_on_file`. The incremental training alters the weights
/// after each time it is presented an input pattern, while batch only alters the weights once after
/// it has been presented to all the patterns.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_train_enum {
    /// Standard backpropagation algorithm, where the weights are
    /// updated after each training pattern. This means that the weights are updated many
    /// times during a single epoch. For this reason some problems will train very fast with
    /// this algorithm, while other more advanced problems will not train very well.
    FANN_TRAIN_INCREMENTAL = 0,
    /// Standard backpropagation algorithm, where the weights are updated after calculating the mean
    /// square error for the whole training set. This means that the weights are only updated once
    /// during an epoch. For this reason some problems will train slower with this algorithm. But
    /// since the mean square error is calculated more correctly than in incremental training, some
    /// problems will reach better solutions with this algorithm.
    FANN_TRAIN_BATCH,
    /// A more advanced batch training algorithm which achieves good results
    /// for many problems. The RPROP training algorithm is adaptive, and does therefore not
    /// use the `learning_rate`. Some other parameters can however be set to change the way the
    /// RPROP algorithm works, but it is only recommended for users with insight in how the RPROP
    /// training algorithm works. The RPROP training algorithm is described by
    /// [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the
    /// iRPROP- training algorithm which is described by [Igel and Husken, 2000] which
    /// is a variant of the standard RPROP training algorithm.
    FANN_TRAIN_RPROP,
    /// A more advanced batch training algorithm which achieves good results
    /// for many problems. The quickprop training algorithm uses the `learning_rate` parameter
    /// along with other more advanced parameters, but it is only recommended to change these
    /// advanced parameters for users with insight in how the quickprop training algorithm works.
    /// The quickprop training algorithm is described by [Fahlman, 1988].
    FANN_TRAIN_QUICKPROP,
}

/// The activation functions used for the neurons during training. The activation functions
/// can either be defined for a group of neurons by `fann_set_activation_function_hidden` and
/// `fann_set_activation_function_output`, or it can be defined for a single neuron by
/// `fann_set_activation_function`.
///
/// The steepness of an activation function is defined in the same way by
/// `fann_set_activation_steepness_hidden`, `fann_set_activation_steepness_output` and
/// `fann_set_activation_steepness`.
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
    /// Neuron does not exist or does not have an activation function.
    FANN_NONE = -1,
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
    FANN_ELLIOTT,
    /// Fast (symmetric sigmoid like) activation function defined by David Elliott
    ///
    /// * span: -1 < y < 1
    ///
    /// * y = (x*s) / (1 + |x*s|)
    ///
    /// * d = s*1/((1+|x*s|)*(1+|x*s|))
    FANN_ELLIOTT_SYMMETRIC,
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
    /// Periodical sine activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = sin(x*s)
    ///
    /// * d = s*cos(x*s)
    FANN_SIN_SYMMETRIC,
    /// Periodical cosine activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = cos(x*s)
    ///
    /// * d = s*-sin(x*s)
    FANN_COS_SYMMETRIC,
    /// Periodical sine activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = sin(x*s)/2+0.5
    ///
    /// * d = s*cos(x*s)/2
    FANN_SIN,
    /// Periodical cosine activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = cos(x*s)/2+0.5
    ///
    /// * d = s*-sin(x*s)/2
    FANN_COS,
}

/// Error function used during training.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_errorfunc_enum {
    /// Standard linear error function.
    FANN_ERRORFUNC_LINEAR = 0,
    /// Tanh error function, usually better but can require a lower learning rate. This error
    /// function aggressively targets outputs that differ much from the desired, while not targeting
    /// outputs that only differ a little that much. This activation function is not recommended for
    /// cascade training and incremental training.
    FANN_ERRORFUNC_TANH,
}

/// Stop criteria used during training.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_stopfunc_enum {
    /// Stop criterion is Mean Square Error (MSE) value.
    FANN_STOPFUNC_MSE = 0,
    /// Stop criterion is number of bits that fail. The number of bits means the
    /// number of output neurons which differ more than the bit fail limit
    /// (see `fann_get_bit_fail_limit`, `fann_set_bit_fail_limit`).
    /// The bits are counted in all of the training data, so this number can be higher than
    /// the number of training data.
    FANN_STOPFUNC_BIT,
}

/// Definition of network types used by `fann_get_network_type`.
#[repr(C)]
#[derive(Copy, Clone)]
pub enum fann_nettype_enum {
    /// Each layer only has connections to the next layer.
    FANN_NETTYPE_LAYER = 0,
    /// Each layer has connections to all following layers.
    FANN_NETTYPE_SHORTCUT,
}

/// This callback function can be called during training when using `fann_train_on_data`,
/// `fann_train_on_file` or `fann_cascadetrain_on_data`.
///
/// The callback can be set by using `fann_set_callback` and is very useful for doing custom
/// things during training. It is recommended to use this function when implementing custom
/// training procedures, or when visualizing the training in a GUI etc. The parameters which the
/// callback function takes are the parameters given to `fann_train_on_data`, plus an `epochs`
/// parameter which tells how many epochs the training has taken so far.
///
/// The callback function should return an integer, if the callback function returns -1, the
/// training will terminate.
///
/// Example of a callback function:
///
/// ```
/// extern crate libc;
/// extern crate fann_sys;
///
/// use libc::*;
/// use fann_sys::*;
///
/// extern "C" fn cb(ann: *mut fann,
///                  train: *mut fann_train_data,
///                  max_epochs: c_uint,
///                  epochs_between_reports: c_uint,
///                  desired_error: c_float,
///                  epochs: c_uint) -> c_int {
///     let mut mse = unsafe { fann_get_MSE(ann) };
///     println!("Epochs: {}. MSE: {}. Desired MSE: {}", epochs, mse, desired_error);
///     0
/// }
///
/// fn main() {
///     let test_callback: fann_callback_type = Some(cb);
/// }
/// ```
pub type fann_callback_type = Option<extern "C" fn(ann: *mut fann,
                                                   train: *mut fann_train_data,
                                                   max_epochs: c_uint,
                                                   epochs_between_reports: c_uint,
                                                   desired_error: c_float,
                                                   epochs: c_uint)
                                     -> c_int>;

#[repr(C)]
struct fann_neuron {
    first_con: c_uint,
    last_con: c_uint,
    sum: fann_type,
    value: fann_type,
    activation_steepness: fann_type,
    activation_function: fann_activationfunc_enum,
}

#[repr(C)]
struct fann_layer {
    first_neuron: *mut fann_neuron,
    last_neuron: *mut fann_neuron,
}

/// Structure used to store error-related information, both
/// `fann` and `fann_train_data` can be cast to this type.
///
/// # See also
/// `fann_set_error_log`, `fann_get_errno`
#[repr(C)]
pub struct fann_error {
    errno_f: fann_errno_enum,
    error_log: *mut FILE,
    errstr: *mut c_char,
}

/// The fast artificial neural network (`fann`) structure.
///
/// Data within this structure should never be accessed directly, but only by using the
/// `fann_get_...` and `fann_set_...` functions.
///
/// The fann structure is created using one of the `fann_create_...` functions and each of
/// the functions which operates on the structure takes a `fann` pointer as the first parameter.
///
/// # See also
/// `fann_create_standard`, `fann_destroy`
#[repr(C)]
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

/// Describes a connection between two neurons and its weight.
///
/// # See Also
/// `fann_get_connection_array`, `fann_set_weight_array`
///
/// This structure appears in FANN >= 2.1.0.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct fann_connection {
    /// Unique number used to identify source neuron
    pub from_neuron: c_uint,
    /// Unique number used to identify destination neuron
    pub to_neuron: c_uint,
    /// The numerical value of the weight
    pub weight: fann_type,
}

/// Structure used to store data, for use with training.
///
/// The data inside this structure should never be manipulated directly, but should use some
/// of the supplied training data manipulation functions.
///
/// The training data structure is very useful for storing data during training and testing of a
/// neural network.
///
/// # See also
/// `fann_read_train_from_file`, `fann_train_on_data`, `fann_destroy_train`
#[repr(C)]
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

#[link(name = "fann")]
extern "C" {
    pub static mut fann_default_error_log: *mut FILE;

    /// Change where errors are logged to. Both `fann` and `fann_data` can be
    /// cast to `fann_error`, so this function can be used to set either of these.
    ///
    /// If `log_file` is NULL, no errors will be printed.
    ///
    /// If `errdat` is NULL, the default log will be set. The default log is the log used when
    /// creating `fann` and `fann_data`. This default log will also be the default for all new
    /// structs that are created.
    ///
    /// The default behavior is to log them to `stderr`.
    ///
    /// # See also
    /// `fann_error`
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_set_error_log(errdat: *mut fann_error, log_file: *mut FILE);

    /// Returns the last error number.
    ///
    /// # See also
    /// `fann_errno_enum`, `fann_reset_errno`
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_get_errno(errdat: *const fann_error) -> fann_errno_enum;

    /// Resets the last error number.
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_reset_errno(errdat: *mut fann_error);

    /// Resets the last error string.
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_reset_errstr(errdat: *mut fann_error);

    /// Returns the last error string.
    ///
    /// This function calls `fann_reset_errno` and `fann_reset_errstr`.
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_get_errstr(errdat: *mut fann_error) -> *mut c_char;

    /// Prints the last error to `stderr`.
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_print_error(errdat: *mut fann_error);

    /// Train one iteration with a set of inputs, and a set of desired outputs.
    /// This training is always incremental training (see `fann_train_enum`), since
    /// only one pattern is presented.
    ///
    /// # Parameters
    ///
    /// * `ann`            - The neural network structure
    /// * `input`          - an array of inputs. This array must be exactly `fann_get_num_input`
    ///                      long.
    /// * `desired_output` - an array of desired outputs. This array must be exactly
    ///                      `fann_get_num_output` long.
    ///
    /// # See also
    /// `fann_train_on_data`, `fann_train_epoch`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_train(ann: *mut fann, input: *const fann_type, desired_output: *const fann_type);

    /// Test with a set of inputs, and a set of desired outputs.
    /// This operation updates the mean square error, but does not
    /// change the network in any way.
    ///
    /// # See also
    /// `fann_test_data`, `fann_train`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_test(ann: *mut fann, input: *const fann_type,
                     desired_output: *const fann_type) -> *mut fann_type;

    /// Reads the mean square error from the network.
    ///
    /// This value is calculated during
    /// training or testing, and can therefore sometimes be a bit off if the weights
    /// have been changed since the last calculation of the value.
    ///
    /// # See also
    /// `fann_test_data`
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_get_MSE(ann: *const fann) -> c_float;

    /// The number of fail bits; means the number of output neurons which differ more
    /// than the bit fail limit (see `fann_get_bit_fail_limit`, `fann_set_bit_fail_limit`).
    /// The bits are counted in all of the training data, so this number can be higher than
    /// the number of training data.
    ///
    /// This value is reset by `fann_reset_MSE` and updated by all the same functions which also
    /// update the MSE value (e.g. `fann_test_data`, `fann_train_epoch`)
    ///
    /// # See also
    /// `fann_stopfunc_enum`, `fann_get_MSE`
    ///
    /// This function appears in FANN >= 2.0.0
    pub fn fann_get_bit_fail(ann: *const fann) -> c_uint;

    /// Resets the mean square error from the network.
    ///
    /// This function also resets the number of bits that fail.
    ///
    /// # See also
    /// `fann_get_bit_fail_limit`, `fann_get_MSE`
    ///
    /// This function appears in FANN >= 1.1.0
    pub fn fann_reset_MSE(ann: *mut fann);

    /// Trains on an entire dataset, for a period of time.
    ///
    /// This training uses the training algorithm chosen by `fann_set_training_algorithm`,
    /// and the parameters set for these training algorithms.
    ///
    /// # Parameters
    ///
    /// * `ann`                    - The neural network
    /// * `data`                   - The data that should be used during training
    /// * `max_epochs`             - The maximum number of epochs the training should continue
    /// * `epochs_between_reports` - The number of epochs between printing a status report to
    ///     `stdout`. A value of zero means no reports should be printed.
    /// * `desired_error`          - The desired `fann_get_MSE` or `fann_get_bit_fail`, depending on
    ///     which stop function is chosen by `fann_set_train_stop_function`.
    ///
    /// Instead of printing out reports every `epochs_between_reports`, a callback function can be
    /// called (see `fann_set_callback`).
    ///
    /// # See also
    /// `fann_train_on_file`, `fann_train_epoch`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_train_on_data(ann: *mut fann,
                              data: *const fann_train_data,
                              max_epochs: c_uint,
                              epochs_between_reports: c_uint,
                              desired_error: c_float);

    /// Does the same as `fann_train_on_data`, but reads the training data directly from a file.
    ///
    /// # See also
    /// `fann_train_on_data`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_train_on_file(ann: *mut fann,
                              filename: *const c_char,
                              max_epochs: c_uint,
                              epochs_between_reports: c_uint,
                              desired_error: c_float);

    /// Train one epoch with a set of training data.
    ///
    /// Train one epoch with the training data stored in `data`. One epoch is where all of
    /// the training data is considered exactly once.
    ///
    /// This function returns the MSE error as it is calculated either before or during
    /// the actual training. This is not the actual MSE after the training epoch, but since
    /// calculating this will require to go through the entire training set once more, it is
    /// more than adequate to use this value during training.
    ///
    /// The training algorithm used by this function is chosen by the `fann_set_training_algorithm`
    /// function.
    ///
    /// # See also
    /// `fann_train_on_data`, `fann_test_data`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_train_epoch(ann: *mut fann, data: *const fann_train_data) -> c_float;

    /// Tests a set of training data and calculates the MSE for the training data.
    ///
    /// This function updates the MSE and the bit fail values.
    ///
    /// # See also
    /// `fann_test`, `fann_get_MSE`, `fann_get_bit_fail`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_test_data(ann: *mut fann, data: *const fann_train_data) -> c_float;

    /// Reads a file that stores training data.
    ///
    /// The file must be formatted like:
    ///
    /// ```text
    /// num_train_data num_input num_output
    /// inputdata separated by space
    /// outputdata separated by space
    /// .
    /// .
    /// .
    /// inputdata separated by space
    /// outputdata separated by space
    /// ```
    ///
    /// # See also
    /// `fann_train_on_data`, `fann_destroy_train`, `fann_save_train`
    ///
    /// This function appears in FANN >= 1.0.0
    pub fn fann_read_train_from_file(filename: *const c_char) -> *mut fann_train_data;

    ///  Creates the training data struct from a user supplied function.
    ///  As the training data are numerable (data 1, data 2...), the user must write
    ///  a function that receives the number of the training data set (input,output)
    ///  and returns the set.
    ///
    ///  # Parameters
    ///
    ///  * `num_data`      - The number of training data
    ///  * `num_input`     - The number of inputs per training data
    ///  * `num_output`    - The number of ouputs per training data
    ///  * `user_function` - The user supplied function
    ///
    ///  # Parameters for the user function
    ///
    ///  * `num`        - The number of the training data set
    ///  * `num_input`  - The number of inputs per training data
    ///  * `num_output` - The number of ouputs per training data
    ///  * `input`      - The set of inputs
    ///  * `output`     - The set of desired outputs
    ///
    /// # See also
    /// `fann_read_train_from_file`, `fann_train_on_data`, `fann_destroy_train`, `fann_save_train`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_create_train_from_callback(num_data: c_uint, num_input: c_uint, num_output: c_uint,
        user_function: Option<extern "C" fn(num: c_uint, num_input: c_uint, num_output: c_uint,
                                            input: *mut fann_type, output: *mut fann_type)>)
        -> *const fann_train_data;

    /// Destructs the training data and properly deallocates all of the associated data.
    /// Be sure to call this function when finished using the training data.
    ///
    /// This function appears in FANN >= 1.0.0
    pub fn fann_destroy_train(train_data: *mut fann_train_data);

    /// Shuffles training data, randomizing the order.
    /// This is recommended for incremental training, while it has no influence during batch
    /// training.
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_shuffle_train_data(train_data: *mut fann_train_data);

    /// Scale input and output data based on previously calculated parameters.
    ///
    /// # Parameters
    ///
    /// * `ann`      - ANN for which trained parameters were calculated before
    /// * `data`     - training data that needs to be scaled
    ///
    /// # See also
    /// `fann_descale_train`, `fann_set_scaling_params`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_scale_train(ann: *mut fann, data: *mut fann_train_data);

    /// Descale input and output data based on previously calculated parameters.
    ///
    /// # Parameters
    ///
    /// * `ann`      - ann for which trained parameters were calculated before
    /// * `data`     - training data that needs to be descaled
    ///
    /// # See also
    /// `fann_scale_train`, `fann_set_scaling_params`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_descale_train(ann: *mut fann, data: *mut fann_train_data);

    /// Calculate input scaling parameters for future use based on training data.
    ///
    /// # Parameters
    ///
    /// * `ann`           - ANN for which parameters need to be calculated
    /// * `data`          - training data that will be used to calculate scaling parameters
    /// * `new_input_min` - desired lower bound in input data after scaling (not strictly followed)
    /// * `new_input_max` - desired upper bound in input data after scaling (not strictly followed)
    ///
    /// # See also
    /// `fann_set_output_scaling_params`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_set_input_scaling_params(ann: *mut fann,
                                         data: *const fann_train_data,
                                         new_input_min: c_float,
                                         new_input_max: c_float) -> c_int;

    /// Calculate output scaling parameters for future use based on training data.
    ///
    /// # Parameters
    ///
    /// * `ann`            - ANN for which parameters need to be calculated
    /// * `data`           - training data that will be used to calculate scaling parameters
    /// * `new_output_min` - desired lower bound in output data after scaling (not strictly
    ///     followed)
    /// * `new_output_max` - desired upper bound in output data after scaling (not strictly
    ///     followed)
    ///
    /// # See also
    /// `fann_set_input_scaling_params`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_set_output_scaling_params(ann: *mut fann,
                                          data: *const fann_train_data,
                                          new_output_min: c_float,
                                          new_output_max: c_float) -> c_int;

    /// Calculate input and output scaling parameters for future use based on training data.
    ///
    /// # Parameters
    ///
    /// * `ann`            - ANN for which parameters need to be calculated
    /// * `data`           - training data that will be used to calculate scaling parameters
    /// * `new_input_min`  - desired lower bound in input data after scaling (not strictly followed)
    /// * `new_input_max`  - desired upper bound in input data after scaling (not strictly followed)
    /// * `new_output_min` - desired lower bound in output data after scaling (not strictly
    ///     followed)
    /// * `new_output_max` - desired upper bound in output data after scaling (not strictly
    ///     followed)
    ///
    /// # See also
    /// `fann_set_input_scaling_params`, `fann_set_output_scaling_params`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_set_scaling_params(ann: *mut fann,
                                   data: *const fann_train_data,
                                   new_input_min: c_float,
                                   new_input_max: c_float,
                                   new_output_min: c_float,
                                   new_output_max: c_float) -> c_int;

    /// Clears scaling parameters.
    ///
    /// # Parameters
    ///
    /// * `ann` - ann for which to clear scaling parameters
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_clear_scaling_params(ann: *mut fann) -> c_int;

    /// Scale data in input vector before feeding it to the ANN based on previously calculated
    /// parameters.
    ///
    /// # Parameters
    ///
    /// `ann`          - for which scaling parameters were calculated
    /// `input_vector` - input vector that will be scaled
    ///
    /// # See also
    /// `fann_descale_input`, `fann_scale_output`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_scale_input(ann: *mut fann, input_vector: *mut fann_type);

    /// Scale data in output vector before feeding it to the ANN based on previously calculated
    /// parameters.
    ///
    /// # Parameters
    ///
    /// * `ann`           - for which scaling parameters were calculated
    /// * `output_vector` - output vector that will be scaled
    ///
    /// # See also
    /// `fann_descale_output`, `fann_scale_intput`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_scale_output(ann: *mut fann, output_vector: *mut fann_type);

    /// Scale data in input vector after getting it from the ANN based on previously calculated
    /// parameters.
    ///
    /// # Parameters
    ///
    /// * `ann`          - for which scaling parameters were calculated
    /// * `input_vector` - input vector that will be descaled
    ///
    /// # See also
    /// `fann_scale_input`, `fann_descale_output`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_descale_input(ann: *mut fann, input_vector: *mut fann_type);

    /// Scale data in output vector after getting it from the ANN based on previously calculated
    /// parameters.
    ///
    /// # Parameters
    ///
    /// * `ann`           - for which scaling parameters were calculated
    /// * `output_vector` - output vector that will be descaled
    ///
    /// # See also
    /// `fann_descale_input`, `fann_scale_output`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_descale_output(ann: *mut fann, output_vector: *mut fann_type);

    /// Scales the inputs in the training data to the specified range.
    ///
    /// # See also
    /// `fann_scale_output_train_data`, `fann_scale_train_data`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_scale_input_train_data(train_data: *mut fann_train_data,
                                       new_min: fann_type, new_max: fann_type);

    /// Scales the outputs in the training data to the specified range.
    ///
    /// # See also
    /// `fann_scale_input_train_data`, `fann_scale_train_data`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_scale_output_train_data(train_data: *mut fann_train_data,
                                        new_min: fann_type, new_max: fann_type);

    /// Scales the inputs and outputs in the training data to the specified range.
    ///
    /// # See also
    /// `fann_scale_output_train_data`, `fann_scale_input_train_data`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_scale_train_data(train_data: *mut fann_train_data,
                                 new_min: fann_type, new_max: fann_type);

    /// Merges the data from `data1` and `data2` into a new `fann_train_data`.
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_merge_train_data(data1: *const fann_train_data,
                                 data2: *const fann_train_data) -> *mut fann_train_data;

    /// Returns an exact copy of a `fann_train_data`.
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_duplicate_train_data(data: *const fann_train_data) -> *mut fann_train_data;

    /// Returns an copy of a subset of the `fann_train_data`, starting at position `pos`
    /// and `length` elements forward.
    ///
    /// ```notest
    /// fann_subset_train_data(train_data, 0, fann_length_train_data(train_data))
    /// ```
    ///
    /// will do the same as `fann_duplicate_train_data`.
    ///
    /// # See also
    /// `fann_length_train_data`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_subset_train_data(data: *const fann_train_data, pos: c_uint, length: c_uint)
        -> *mut fann_train_data;

    /// Returns the number of training patterns in the `fann_train_data`.
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_length_train_data(data: *const fann_train_data) -> c_uint;

    /// Returns the number of inputs in each of the training patterns in the `fann_train_data`.
    ///
    /// # See also
    /// `fann_num_train_data`, `fann_num_output_train_data`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_num_input_train_data(data: *const fann_train_data) -> c_uint;

    /// Returns the number of outputs in each of the training patterns in the `fann_train_data`.
    ///
    /// # See also
    /// `fann_num_train_data`, `fann_num_input_train_data`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_num_output_train_data(data: *const fann_train_data) -> c_uint;

    /// Save the training structure to a file, with the format specified in
    /// `fann_read_train_from_file`
    ///
    /// # Return
    ///
    /// The function returns 0 on success and -1 on failure.
    ///
    /// # See also
    /// `fann_read_train_from_file`, `fann_save_train_to_fixed`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_save_train(data: *mut fann_train_data, filename: *const c_char) -> c_int;

    /// Saves the training structure to a fixed point data file.
    ///
    /// This function is very useful for testing the quality of a fixed point network.
    ///
    /// # Return
    ///
    /// The function returns 0 on success and -1 on failure.
    ///
    /// # See also
    /// `fann_save_train`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_save_train_to_fixed(data: *mut fann_train_data,
                                    filename: *const c_char,
                                    decimal_point: c_uint) -> c_int;

    /// Return the training algorithm as described by `fann_train_enum`. This training algorithm
    /// is used by `fann_train_on_data` and associated functions.
    ///
    /// Note that this algorithm is also used during `fann_cascadetrain_on_data`, although only
    /// `FANN_TRAIN_RPROP` and `FANN_TRAIN_QUICKPROP` is allowed during cascade training.
    ///
    /// The default training algorithm is `FANN_TRAIN_RPROP`.
    ///
    /// # See also
    /// `fann_set_training_algorithm`, `fann_train_enum`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_get_training_algorithm(ann: *const fann) -> fann_train_enum;

    /// Set the training algorithm.
    ///
    /// More info available in `fann_get_training_algorithm`.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_set_training_algorithm(ann: *mut fann, training_algorithm: fann_train_enum);

    /// Return the learning rate.
    ///
    /// The learning rate is used to determine how aggressive training should be for some of the
    /// training algorithms (`FANN_TRAIN_INCREMENTAL`, `FANN_TRAIN_BATCH`, `FANN_TRAIN_QUICKPROP`).
    /// Do however note that it is not used in `FANN_TRAIN_RPROP`.
    ///
    /// The default learning rate is 0.7.
    ///
    /// # See also
    /// `fann_set_learning_rate`, `fann_set_training_algorithm`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_get_learning_rate(ann: *const fann) -> c_float;

    /// Set the learning rate.
    ///
    /// More info available in `fann_get_learning_rate`.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_set_learning_rate(ann: *mut fann, learning_rate: c_float);

    /// Get the learning momentum.
    ///
    /// The learning momentum can be used to speed up FANN_TRAIN_INCREMENTAL training.
    /// A too high momentum will however not benefit training. Setting momentum to 0 will
    /// be the same as not using the momentum parameter. The recommended value of this parameter
    /// is between 0.0 and 1.0.
    ///
    /// The default momentum is 0.
    ///
    /// # See also
    /// `fann_set_learning_momentum`, `fann_set_training_algorithm`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_learning_momentum(ann: *const fann) -> c_float;

    /// Set the learning momentum.
    ///
    /// More info available in `fann_get_learning_momentum`.
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_learning_momentum(ann: *mut fann, learning_momentum: c_float);

    /// Get the activation function for neuron number `neuron` in layer number `layer`,
    /// counting the input layer as layer 0.
    ///
    /// It is not possible to get activation functions for the neurons in the input layer.
    ///
    /// Information about the individual activation functions is available at
    /// `fann_activationfunc_enum`.
    ///
    /// # Returns
    ///
    /// The activation function for the neuron or `FANN_NONE` if the neuron is not defined in the
    /// neural network.
    ///
    /// # See also
    /// `fann_set_activation_function_layer`, `fann_set_activation_function_hidden`,
    /// `fann_set_activation_function_output`, `fann_set_activation_steepness`,
    /// `fann_set_activation_function`
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_get_activation_function(ann: *const fann, layer: c_int, neuron: c_int)
        -> fann_activationfunc_enum;

    /// Set the activation function for neuron number `neuron` in layer number `layer`,
    /// counting the input layer as layer 0.
    ///
    /// It is not possible to set activation functions for the neurons in the input layer.
    ///
    /// When choosing an activation function it is important to note that the activation
    /// functions have different range. `FANN_SIGMOID` is e.g. in the 0 - 1 range while
    /// `FANN_SIGMOID_SYMMETRIC` is in the -1 - 1 range and `FANN_LINEAR` is unbounded.
    ///
    /// Information about the individual activation functions is available at
    /// `fann_activationfunc_enum`.
    ///
    /// The default activation function is `FANN_SIGMOID_STEPWISE`.
    ///
    /// # See also
    /// `fann_set_activation_function_layer`, `fann_set_activation_function_hidden`,
    /// `fann_set_activation_function_output`, `fann_set_activation_steepness`,
    /// `fann_get_activation_function`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_activation_function(ann: *mut fann,
                                        activation_function: fann_activationfunc_enum,
                                        layer: c_int,
                                        neuron: c_int);

    /// Set the activation function for all the neurons in the layer number `layer`,
    /// counting the input layer as layer 0.
    ///
    /// It is not possible to set activation functions for the neurons in the input layer.
    ///
    /// # See also
    /// `fann_set_activation_function`, `fann_set_activation_function_hidden`,
    /// `fann_set_activation_function_output`, `fann_set_activation_steepness_layer`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_activation_function_layer(ann: *mut fann,
                                              activation_function: fann_activationfunc_enum,
                                              layer: c_int);

    /// Set the activation function for all of the hidden layers.
    ///
    /// # See also
    /// `fann_set_activation_function`, `fann_set_activation_function_layer`,
    /// `fann_set_activation_function_output`, `fann_set_activation_steepness_hidden`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_set_activation_function_hidden(ann: *mut fann,
                                               activation_function: fann_activationfunc_enum);

    /// Set the activation function for the output layer.
    ///
    /// # See also
    /// `fann_set_activation_function`, `fann_set_activation_function_layer`,
    /// `fann_set_activation_function_hidden`, `fann_set_activation_steepness_output`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_set_activation_function_output(ann: *mut fann,
                                               activation_function: fann_activationfunc_enum);

    /// Get the activation steepness for neuron number `neuron` in layer number `layer`,
    /// counting the input layer as layer 0.
    ///
    /// It is not possible to get activation steepness for the neurons in the input layer.
    ///
    /// The steepness of an activation function says something about how fast the activation
    /// function goes from the minimum to the maximum. A high value for the activation function will
    /// also give a more aggressive training.
    ///
    /// When training neural networks where the output values should be at the extremes (usually 0
    /// and 1, depending on the activation function), a steep activation function can be used (e.g.
    /// 1.0).
    ///
    /// The default activation steepness is 0.5.
    ///
    /// # Returns
    /// The activation steepness for the neuron or -1 if the neuron is not defined in the neural
    /// network.
    ///
    /// #See also
    /// `fann_set_activation_steepness_layer`, `fann_set_activation_steepness_hidden`,
    /// `fann_set_activation_steepness_output`, `fann_set_activation_function`,
    /// `fann_set_activation_steepness`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_get_activation_steepness(ann: *const fann, layer: c_int, neuron: c_int)
        -> fann_type;

    /// Set the activation steepness for neuron number `neuron` in layer number `layer`,
    /// counting the input layer as layer 0.
    ///
    /// It is not possible to set activation steepness for the neurons in the input layer.
    ///
    /// The steepness of an activation function says something about how fast the activation
    /// function goes from the minimum to the maximum. A high value for the activation function will
    /// also give a more aggressive training.
    ///
    /// When training neural networks where the output values should be at the extremes (usually 0
    /// and 1, depending on the activation function), a steep activation function can be used (e.g.
    /// 1.0).
    ///
    /// The default activation steepness is 0.5.
    ///
    /// # See also
    /// `fann_set_activation_steepness_layer`, `fann_set_activation_steepness_hidden`,
    /// `fann_set_activation_steepness_output`, `fann_set_activation_function`,
    /// `fann_get_activation_steepness`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_activation_steepness(ann: *mut fann,
                                         steepness: fann_type,
                                         layer: c_int,
                                         neuron: c_int);

    /// Set the activation steepness for all neurons in layer number `layer`,
    /// counting the input layer as layer 0.
    ///
    /// It is not possible to set activation steepness for the neurons in the input layer.
    ///
    /// # See also
    /// `fann_set_activation_steepness`, `fann_set_activation_steepness_hidden`,
    /// `fann_set_activation_steepness_output`, `fann_set_activation_function_layer`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_activation_steepness_layer(ann: *mut fann, steepness: fann_type, layer: c_int);

    /// Set the steepness of the activation steepness in all of the hidden layers.
    ///
    /// See also:
    /// `fann_set_activation_steepness`, `fann_set_activation_steepness_layer`,
    /// `fann_set_activation_steepness_output`, `fann_set_activation_function_hidden`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_activation_steepness_hidden(ann: *mut fann, steepness: fann_type);

    /// Set the steepness of the activation steepness in the output layer.
    ///
    /// # See also
    /// `fann_set_activation_steepness`, `fann_set_activation_steepness_layer`,
    /// `fann_set_activation_steepness_hidden`, `fann_set_activation_function_output`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_activation_steepness_output(ann: *mut fann, steepness: fann_type);

    /// Returns the error function used during training.
    ///
    /// The error functions are described further in `fann_errorfunc_enum`.
    ///
    /// The default error function is `FANN_ERRORFUNC_TANH`
    ///
    /// # See also
    /// `fann_set_train_error_function`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_get_train_error_function(ann: *const fann) -> fann_errorfunc_enum;

    /// Set the error function used during training.
    ///
    /// The error functions are described further in `fann_errorfunc_enum`.
    ///
    /// # See also
    /// `fann_get_train_error_function`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_train_error_function(ann: *mut fann,
                                         train_error_function: fann_errorfunc_enum);

    /// Returns the the stop function used during training.
    ///
    /// The stop function is described further in `fann_stopfunc_enum`.
    ///
    /// The default stop function is `FANN_STOPFUNC_MSE`.
    ///
    /// # See also
    /// `fann_get_train_stop_function`, `fann_get_bit_fail_limit`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_train_stop_function(ann: *const fann) -> fann_stopfunc_enum;

    /// Set the stop function used during training.
    ///
    /// Returns the the stop function used during training.
    ///
    /// The stop function is described further in `fann_stopfunc_enum`.
    ///
    /// # See also
    /// `fann_get_train_stop_function`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_train_stop_function(ann: *mut fann,
                                        train_stop_function: fann_stopfunc_enum);

    /// Returns the bit fail limit used during training.
    ///
    /// The bit fail limit is used during training where the `fann_stopfunc_enum` is set to
    /// `FANN_STOPFUNC_BIT`.
    ///
    /// The limit is the maximum accepted difference between the desired output and the actual
    /// output during training. Each output that diverges more than this limit is counted as an
    /// error bit. This difference is divided by two when dealing with symmetric activation
    /// functions, so that symmetric and not symmetric activation functions can use the same limit.
    ///
    /// The default bit fail limit is 0.35.
    ///
    /// # See also
    /// `fann_set_bit_fail_limit`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_bit_fail_limit(ann: *const fann) -> fann_type;

    /// Set the bit fail limit used during training.
    ///
    /// # See also
    /// `fann_get_bit_fail_limit`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_bit_fail_limit(ann: *mut fann, bit_fail_limit: fann_type);

    /// Sets the callback function for use during training.
    ///
    /// See `fann_callback_type` for more information about the callback function.
    ///
    /// The default callback function simply prints out some status information.
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_callback(ann: *mut fann, callback: fann_callback_type);

    /// The decay is a small negative valued number which is the factor that the weights
    /// should become smaller in each iteration during quickprop training. This is used
    /// to make sure that the weights do not become too high during training.
    ///
    /// The default decay is -0.0001.
    ///
    /// # See also
    /// `fann_set_quickprop_decay`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_get_quickprop_decay(ann: *const fann) -> c_float;

    /// Sets the quickprop decay factor.
    ///
    /// # See also
    /// `fann_get_quickprop_decay`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_quickprop_decay(ann: *mut fann, quickprop_decay: c_float);

    /// The mu factor is used to increase and decrease the step size during quickprop training.
    /// The mu factor should always be above 1, since it would otherwise decrease the step size
    /// when it was supposed to increase it.
    ///
    /// The default mu factor is 1.75.
    ///
    /// # See also
    /// `fann_set_quickprop_mu`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_get_quickprop_mu(ann: *const fann) -> c_float;

    /// Sets the quickprop mu factor.
    ///
    /// # See also
    /// `fann_get_quickprop_mu`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_quickprop_mu(ann: *mut fann, quickprop_mu: c_float);

    /// The increase factor is a value larger than 1, which is used to
    /// increase the step size during RPROP training.
    ///
    /// The default increase factor is 1.2.
    ///
    /// # See also
    /// `fann_set_rprop_increase_factor`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_get_rprop_increase_factor(ann: *const fann) -> c_float;

    /// The increase factor used during RPROP training.
    ///
    /// # See also
    /// `fann_get_rprop_increase_factor`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_rprop_increase_factor(ann: *mut fann, rprop_increase_factor: c_float);

    /// The decrease factor is a value smaller than 1, which is used to decrease the step size
    /// during RPROP training.
    ///
    /// The default decrease factor is 0.5.
    ///
    /// # See also
    /// `fann_set_rprop_decrease_factor`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_get_rprop_decrease_factor(ann: *const fann) -> c_float;

    /// The decrease factor is a value smaller than 1, which is used to decrease the step size
    /// during RPROP training.
    ///
    /// # See also
    /// `fann_get_rprop_decrease_factor`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_rprop_decrease_factor(ann: *mut fann, rprop_decrease_factor: c_float);

    /// The minimum step size is a small positive number determining how small the minimum step size
    /// may be.
    ///
    /// The default value delta min is 0.0.
    ///
    /// # See also
    /// `fann_set_rprop_delta_min`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_get_rprop_delta_min(ann: *const fann) -> c_float;

    /// The minimum step size is a small positive number determining how small the minimum step size
    /// may be.
    ///
    /// # See also
    /// `fann_get_rprop_delta_min`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_rprop_delta_min(ann: *mut fann, rprop_delta_min: c_float);

    /// The maximum step size is a positive number determining how large the maximum step size may
    /// be.
    ///
    /// The default delta max is 50.0.
    ///
    /// # See also
    /// `fann_set_rprop_delta_max`, `fann_get_rprop_delta_min`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_get_rprop_delta_max(ann: *const fann) -> c_float;

    /// The maximum step size is a positive number determining how large the maximum step size may
    /// be.
    ///
    /// # See also
    /// `fann_get_rprop_delta_max`, `fann_get_rprop_delta_min`
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_set_rprop_delta_max(ann: *mut fann, rprop_delta_max: c_float);

    /// The initial step size is a positive number determining the initial step size.
    ///
    /// The default delta zero is 0.1.
    ///
    /// # See also
    /// `fann_set_rprop_delta_zero`, `fann_get_rprop_delta_min`, `fann_get_rprop_delta_max`
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_get_rprop_delta_zero(ann: *const fann) -> c_float;

    /// The initial step size is a positive number determining the initial step size.
    ///
    /// # See also
    /// `fann_get_rprop_delta_zero`, `fann_get_rprop_delta_zero`
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_set_rprop_delta_zero(ann: *mut fann, rprop_delta_max: c_float);

    /// Trains on an entire dataset, for a period of time using the Cascade2 training algorithm.
    /// This algorithm adds neurons to the neural network while training, which means that it
    /// needs to start with an ANN without any hidden layers. The neural network should also use
    /// shortcut connections, so `fann_create_shortcut` should be used to create the ANN like this:
    ///
    /// ```notest
    /// let ann = fann_create_shortcut(2,
    ///                                fann_num_input_train_data(train_data),
    ///                                fann_num_output_train_data(train_data));
    /// ```
    ///
    /// This training uses the parameters set using `fann_set_cascade_...`, but it also uses
    /// another training algorithm as it's internal training algorithm. This algorithm can be set to
    /// either `FANN_TRAIN_RPROP` or `FANN_TRAIN_QUICKPROP` by `fann_set_training_algorithm`, and
    /// the parameters set for these training algorithms will also affect the cascade training.
    ///
    /// # Parameters
    ///
    /// * `ann`                     - The neural network
    /// * `data`                    - The data that should be used during training
    /// * `max_neuron`              - The maximum number of neurons to be added to the ANN
    /// * `neurons_between_reports` - The number of neurons between printing a status report to
    ///     stdout. A value of zero means no reports should be printed.
    /// * `desired_error`           - The desired `fann_get_MSE` or `fann_get_bit_fail`, depending
    ///     on which stop function is chosen by `fann_set_train_stop_function`.
    ///
    /// Instead of printing out reports every neurons_between_reports, a callback function can be
    /// called (see `fann_set_callback`).
    ///
    /// # See also
    /// `fann_train_on_data`, `fann_cascadetrain_on_file`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_cascadetrain_on_data(ann: *mut fann,
                                     data: *const fann_train_data,
                                     max_neurons: c_uint,
                                     neurons_between_reports: c_uint,
                                     desired_error: c_float);

    /// Does the same as `fann_cascadetrain_on_data`, but reads the training data directly from a
    /// file.
    ///
    /// # See also
    /// `fann_cascadetrain_on_data`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_cascadetrain_on_file(ann: *mut fann,
                                     filename: *const c_char,
                                     max_neurons: c_uint,
                                     neurons_between_reports: c_uint,
                                     desired_error: c_float);

    /// The cascade output change fraction is a number between 0 and 1 determining how large a
    /// fraction the `fann_get_MSE` value should change within
    /// `fann_get_cascade_output_stagnation_epochs` during training of the output connections, in
    /// order for the training not to stagnate. If the training stagnates, the training of the
    /// output connections will be ended and new candidates will be prepared.
    ///
    /// This means:
    /// If the MSE does not change by a fraction of `fann_get_cascade_output_change_fraction` during
    /// a period of `fann_get_cascade_output_stagnation_epochs`, the training of the output
    /// connections is stopped because the training has stagnated.
    ///
    /// If the cascade output change fraction is low, the output connections will be trained more
    /// and if the fraction is high they will be trained less.
    ///
    /// The default cascade output change fraction is 0.01, which is equivalent to a 1% change in
    /// MSE.
    ///
    /// # See also
    /// `fann_set_cascade_output_change_fraction`, `fann_get_MSE`,
    /// `fann_get_cascade_output_stagnation_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_output_change_fraction(ann: *const fann) -> c_float;

    /// Sets the cascade output change fraction.
    ///
    /// # See also
    /// `fann_get_cascade_output_change_fraction`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_output_change_fraction(ann: *mut fann,
                                                   cascade_output_change_fraction: c_float);

    /// The number of cascade output stagnation epochs determines the number of epochs training is
    /// allowed to continue without changing the MSE by a fraction of
    /// `fann_get_cascade_output_change_fraction`.
    ///
    /// See more info about this parameter in `fann_get_cascade_output_change_fraction`.
    ///
    /// The default number of cascade output stagnation epochs is 12.
    ///
    /// # See also
    /// `fann_set_cascade_output_stagnation_epochs`, `fann_get_cascade_output_change_fraction`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_output_stagnation_epochs(ann: *const fann) -> c_uint;

    /// Sets the number of cascade output stagnation epochs.
    ///
    /// # See also
    /// `fann_get_cascade_output_stagnation_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_output_stagnation_epochs(ann: *mut fann,
                                                     cascade_output_stagnation_epochs: c_uint);

    /// The cascade candidate change fraction is a number between 0 and 1 determining how large a
    /// fraction the `fann_get_MSE` value should change within
    /// `fann_get_cascade_candidate_stagnation_epochs` during training of the candidate neurons, in
    /// order for the training not to stagnate. If the training stagnates, the training of the
    /// candidate neurons will be ended and the best candidate will be selected.
    ///
    /// This means:
    /// If the MSE does not change by a fraction of `fann_get_cascade_candidate_change_fraction`
    /// during a period of `fann_get_cascade_candidate_stagnation_epochs`, the training of the
    /// candidate neurons is stopped because the training has stagnated.
    ///
    /// If the cascade candidate change fraction is low, the candidate neurons will be trained more
    /// and if the fraction is high they will be trained less.
    ///
    /// The default cascade candidate change fraction is 0.01, which is equivalent to a 1% change in
    /// MSE.
    ///
    /// # See also
    /// `fann_set_cascade_candidate_change_fraction`, `fann_get_MSE`,
    /// `fann_get_cascade_candidate_stagnation_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_candidate_change_fraction(ann: *const fann) -> c_float;

    /// Sets the cascade candidate change fraction.
    ///
    /// # See also
    /// `fann_get_cascade_candidate_change_fraction`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_candidate_change_fraction(ann: *mut fann,
                                                      cascade_candidate_change_fraction: c_float);

    /// The number of cascade candidate stagnation epochs determines the number of epochs training
    /// is allowed to continue without changing the MSE by a fraction of
    /// `fann_get_cascade_candidate_change_fraction`.
    ///
    /// See more info about this parameter in `fann_get_cascade_candidate_change_fraction`.
    ///
    /// The default number of cascade candidate stagnation epochs is 12.
    ///
    /// # See also
    /// `fann_set_cascade_candidate_stagnation_epochs`, `fann_get_cascade_candidate_change_fraction`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_candidate_stagnation_epochs(ann: *const fann) -> c_uint;

    /// Sets the number of cascade candidate stagnation epochs.
    ///
    /// # See also
    /// `fann_get_cascade_candidate_stagnation_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_candidate_stagnation_epochs(ann: *mut fann,
        cascade_candidate_stagnation_epochs: c_uint);

    /// The weight multiplier is a parameter which is used to multiply the weights from the
    /// candidate neuron before adding the neuron to the neural network. This parameter is usually
    /// between 0 and 1, and is used to make the training a bit less aggressive.
    ///
    /// The default weight multiplier is 0.4
    ///
    /// # See also
    /// `fann_set_cascade_weight_multiplier`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_weight_multiplier(ann: *const fann) -> fann_type;

    /// Sets the weight multiplier.
    ///
    /// # See also
    /// `fann_get_cascade_weight_multiplier`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_weight_multiplier(ann: *mut fann,
                                              cascade_weight_multiplier: fann_type);

    /// The candidate limit is a limit for how much the candidate neuron may be trained.
    /// The limit is a limit on the proportion between the MSE and candidate score.
    ///
    /// Set this to a lower value to avoid overfitting and to a higher if overfitting is
    /// not a problem.
    ///
    /// The default candidate limit is 1000.0
    ///
    /// # See also
    /// `fann_set_cascade_candidate_limit`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_candidate_limit(ann: *const fann) -> fann_type;

    /// Sets the candidate limit.
    ///
    /// # See also
    /// `fann_get_cascade_candidate_limit`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_candidate_limit(ann: *mut fann,
                                            cascade_candidate_limit: fann_type);

    /// The maximum out epochs determines the maximum number of epochs the output connections
    /// may be trained after adding a new candidate neuron.
    ///
    /// The default max out epochs is 150
    ///
    /// # See also
    /// `fann_set_cascade_max_out_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_max_out_epochs(ann: *const fann) -> c_uint;

    /// Sets the maximum out epochs.
    ///
    /// # See also
    /// `fann_get_cascade_max_out_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_max_out_epochs(ann: *mut fann, cascade_max_out_epochs: c_uint);

    /// The maximum candidate epochs determines the maximum number of epochs the input
    /// connections to the candidates may be trained before adding a new candidate neuron.
    ///
    /// The default max candidate epochs is 150.
    ///
    /// # See also
    /// `fann_set_cascade_max_cand_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_max_cand_epochs(ann: *const fann) -> c_uint;

    /// Sets the max candidate epochs.
    ///
    /// # See also
    /// `fann_get_cascade_max_cand_epochs`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_max_cand_epochs(ann: *mut fann,
                                            cascade_max_cand_epochs: c_uint);

    /// The number of candidates used during training (calculated by multiplying
    /// `fann_get_cascade_activation_functions_count`,
    /// `fann_get_cascade_activation_steepnesses_count` and
    /// `fann_get_cascade_num_candidate_groups`).
    ///
    /// The actual candidates is defined by the `fann_get_cascade_activation_functions` and
    /// `fann_get_cascade_activation_steepnesses` arrays. These arrays define the activation
    /// functions and activation steepnesses used for the candidate neurons. If there are 2
    /// activation functions in the activation function array and 3 steepnesses in the steepness
    /// array, then there will be 2x3=6 different candidates which will be trained. These 6
    /// different candidates can be copied into several candidate groups, where the only difference
    /// between these groups is the initial weights. If the number of groups is set to 2, then the
    /// number of candidate neurons will be 2x3x2=12. The number of candidate groups is defined by
    /// `fann_set_cascade_num_candidate_groups`.
    ///
    /// The default number of candidates is 6x4x2 = 48
    ///
    /// # See also
    /// `fann_get_cascade_activation_functions`, `fann_get_cascade_activation_functions_count`,
    /// `fann_get_cascade_activation_steepnesses`, `fann_get_cascade_activation_steepnesses_count`,
    /// `fann_get_cascade_num_candidate_groups`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_num_candidates(ann: *const fann) -> c_uint;

    /// The number of activation functions in the `fann_get_cascade_activation_functions` array.
    ///
    /// The default number of activation functions is 6.
    ///
    /// # See also
    /// `fann_get_cascade_activation_functions`, `fann_set_cascade_activation_functions`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_activation_functions_count(ann: *const fann) -> c_uint;

    /// The cascade activation functions array is an array of the different activation functions
    /// used by the candidates.
    ///
    /// See `fann_get_cascade_num_candidates` for a description of which candidate neurons will be
    /// generated by this array.
    ///
    /// The default activation functions is {`FANN_SIGMOID`, `FANN_SIGMOID_SYMMETRIC`,
    /// `FANN_GAUSSIAN`, `FANN_GAUSSIAN_SYMMETRIC`, `FANN_ELLIOTT`, `FANN_ELLIOTT_SYMMETRIC`,
    /// `FANN_SIN_SYMMETRIC`, `FANN_COS_SYMMETRIC`, `FANN_SIN`, `FANN_COS`}
    ///
    /// # See also
    /// `fann_get_cascade_activation_functions_count`, `fann_set_cascade_activation_functions`,
    /// `fann_activationfunc_enum`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_activation_functions(ann: *const fann) -> *mut fann_activationfunc_enum;

    /// Sets the array of cascade candidate activation functions. The array must be just as long
    /// as defined by the count.
    ///
    /// See `fann_get_cascade_num_candidates` for a description of which candidate neurons will be
    /// generated by this array.
    ///
    /// # See also
    /// `fann_get_cascade_activation_steepnesses_count`, `fann_get_cascade_activation_steepnesses`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_activation_functions(ann: *mut fann,
        cascade_activation_functions: *const fann_activationfunc_enum,
        cascade_activation_functions_count: c_uint);

    /// The number of activation steepnesses in the `fann_get_cascade_activation_functions` array.
    ///
    /// The default number of activation steepnesses is 4.
    ///
    /// # See also
    /// `fann_get_cascade_activation_steepnesses`, `fann_set_cascade_activation_functions`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_activation_steepnesses_count(ann: *const fann) -> c_uint;

    /// The cascade activation steepnesses array is an array of the different activation functions
    /// used by the candidates.
    ///
    /// See `fann_get_cascade_num_candidates` for a description of which candidate neurons will be
    /// generated by this array.
    ///
    /// The default activation steepnesses is {0.25, 0.50, 0.75, 1.00}
    ///
    /// # See also
    /// `fann_set_cascade_activation_steepnesses`, `fann_get_cascade_activation_steepnesses_count`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_activation_steepnesses(ann: *const fann) -> *mut fann_type;

    /// Sets the array of cascade candidate activation steepnesses. The array must be just as long
    /// as defined by the count.
    ///
    /// See `fann_get_cascade_num_candidates` for a description of which candidate neurons will be
    /// generated by this array.
    ///
    /// # See also
    /// `fann_get_cascade_activation_steepnesses`, `fann_get_cascade_activation_steepnesses_count`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_activation_steepnesses(ann: *mut fann,
        cascade_activation_steepnesses: *const fann_type,
        cascade_activation_steepnesses_count: c_uint);

    /// The number of candidate groups is the number of groups of identical candidates which will be
    /// used during training.
    ///
    /// This number can be used to have more candidates without having to define new parameters for
    /// the candidates.
    ///
    /// See `fann_get_cascade_num_candidates` for a description of which candidate neurons will be
    /// generated by this parameter.
    ///
    /// The default number of candidate groups is 2.
    ///
    /// # See also
    /// `fann_set_cascade_num_candidate_groups`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_get_cascade_num_candidate_groups(ann: *const fann) -> c_uint;

    /// Sets the number of candidate groups.
    ///
    /// # See also
    /// `fann_get_cascade_num_candidate_groups`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_set_cascade_num_candidate_groups(ann: *mut fann,
                                                 cascade_num_candidate_groups: c_uint);

    /// Constructs a backpropagation neural network from a configuration file, which has been saved
    /// by `fann_save`.
    ///
    /// # See also
    /// `fann_save`, `fann_save_to_fixed`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_create_from_file(configuration_file: *const c_char) -> *mut fann;

    /// Save the entire network to a configuration file.
    ///
    /// The configuration file contains all information about the neural network and enables
    /// `fann_create_from_file` to create an exact copy of the neural network and all of the
    /// parameters associated with the neural network.
    ///
    /// These three parameters (`fann_set_callback`, `fann_set_error_log`,
    /// `fann_set_user_data`) are *NOT* saved to the file because they cannot safely be
    /// ported to a different location. Also temporary parameters generated during training
    /// like `fann_get_MSE` are not saved.
    ///
    /// # Return
    /// The function returns 0 on success and -1 on failure.
    ///
    /// # See also
    /// `fann_create_from_file`, `fann_save_to_fixed`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_save(ann: *mut fann, configuration_file: *const c_char) -> c_int;


    /// Saves the entire network to a configuration file.
    /// But it is saved in fixed point format no matter which
    /// format it is currently in.
    ///
    /// This is useful for training a network in floating points,
    /// and then later executing it in fixed point.
    ///
    /// The function returns the bit position of the fix point, which
    /// can be used to find out how accurate the fixed point network will be.
    /// A high value indicates high precision, and a low value indicates low
    /// precision.
    ///
    /// A negative value indicates very low precision, and a very strong possibility for overflow.
    /// (the actual fix point will be set to 0, since a negative fix point does not make sense).
    ///
    /// Generally, a fix point lower than 6 is bad, and should be avoided.
    /// The best way to avoid this is to have fewer connections to each neuron,
    /// or just fewer neurons in each layer.
    ///
    /// The fixed point use of this network is only intended for use on machines that
    /// have no floating point processor, like an iPAQ. On normal computers the floating
    /// point version is actually faster.
    ///
    /// # See also
    /// `fann_create_from_file`, `fann_save`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_save_to_fixed(ann: *mut fann, configuration_file: *const c_char) -> c_int;

    /// Creates a standard fully connected backpropagation neural network.
    ///
    /// There will be a bias neuron in each layer (except the output layer),
    /// and this bias neuron will be connected to all neurons in the next layer.
    /// When running the network, the bias nodes always emit 1.
    ///
    /// To destroy a `fann` use the `fann_destroy` function.
    ///
    /// # Parameters
    ///
    /// * `num_layers` - The total number of layers including the input and the output layer.
    /// * `...`        - Integer values determining the number of neurons in each layer starting
    ///     with the input layer and ending with the output layer.
    ///
    /// # Returns
    ///
    /// A pointer to the newly created `fann`.
    ///
    /// # Example
    ///
    ///
    /// ```
    /// // Creating an ANN with 2 input neurons, 1 output neuron,
    /// // and two hidden layers with 8 and 9 neurons
    /// unsafe {
    ///     let ann = fann_sys::fann_create_standard(4, 2, 8, 9, 1);
    /// }
    /// ```
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_create_standard(num_layers: c_uint, ...) -> *mut fann;

    /// Just like `fann_create_standard`, but with an array of layer sizes
    /// instead of individual parameters.
    ///
    /// # Example
    ///
    /// ```
    /// // Creating an ANN with 2 input neurons, 1 output neuron,
    /// // and two hidden layers with 8 and 9 neurons
    /// let layers = [2, 8, 9, 1];
    /// unsafe {
    ///     let ann = fann_sys::fann_create_standard_array(4, layers.as_ptr());
    /// }
    /// ```
    ///
    /// # See also
    /// `fann_create_standard`, `fann_create_sparse`, `fann_create_shortcut`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_create_standard_array(num_layers: c_uint, layers: *const c_uint) -> *mut fann;

    /// Creates a standard backpropagation neural network, which is not fully connected.
    ///
    /// # Parameters
    ///
    /// * `connection_rate` - The connection rate controls how many connections there will be in the
    ///     network. If the connection rate is set to 1, the network will be fully
    ///     connected, but if it is set to 0.5, only half of the connections will be set.
    ///     A connection rate of 1 will yield the same result as `fann_create_standard`.
    /// * `num_layers`      - The total number of layers including the input and the output layer.
    /// * `...`             - Integer values determining the number of neurons in each layer
    ///     starting with the input layer and ending with the output layer.
    ///
    /// # Returns
    /// A pointer to the newly created `fann`.
    ///
    /// # See also
    /// `fann_create_sparse_array`, `fann_create_standard`, `fann_create_shortcut`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_create_sparse(connection_rate: c_float, num_layers: c_uint, ...) -> *mut fann;

    /// Just like `fann_create_sparse`, but with an array of layer sizes
    /// instead of individual parameters.
    ///
    /// See `fann_create_standard_array` for a description of the parameters.
    ///
    /// # See also
    /// `fann_create_sparse`, `fann_create_standard`, `fann_create_shortcut`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_create_sparse_array(connection_rate: c_float,
                                    num_layers: c_uint,
                                    layers: *const c_uint) -> *mut fann;

    /// Creates a standard backpropagation neural network, which is not fully connected and which
    /// also has shortcut connections.
    ///
    /// Shortcut connections are connections that skip layers. A fully connected network with
    /// shortcut connections is a network where all neurons are connected to all neurons in later
    /// layers. Including direct connections from the input layer to the output layer.
    ///
    /// See `fann_create_standard` for a description of the parameters.
    ///
    /// # See also
    /// `fann_create_shortcut_array`, `fann_create_standard`, `fann_create_sparse`,
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_create_shortcut(num_layers: c_uint, ...) -> *mut fann;

    /// Just like `fann_create_shortcut`, but with an array of layer sizes
    /// instead of individual parameters.
    ///
    /// See `fann_create_standard_array` for a description of the parameters.
    ///
    /// # See also
    /// `fann_create_shortcut`, `fann_create_standard`, `fann_create_sparse`
    ///
    /// This function appears in FANN >= 2.0.0.
    pub fn fann_create_shortcut_array(num_layers: c_uint, layers: *const c_uint) -> *mut fann;

    /// Destroys the entire network, properly freeing all the associated memory.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_destroy(ann: *mut fann);

    /// Runs input through the neural network, returning an array of outputs, the number of
    /// which being equal to the number of neurons in the output layer.
    ///
    /// Ownership of the output array remains with the `fann` structure. It may be overwritten by
    /// subsequent function calls. Do not deallocate it!
    ///
    /// # See also
    /// `fann_test`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_run(ann: *mut fann, input: *const fann_type) -> *mut fann_type;

    /// Give each connection a random weight between `min_weight` and `max_weight`.
    ///
    /// From the beginning the weights are random between -0.1 and 0.1.
    ///
    /// # See also
    /// `fann_init_weights`
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_randomize_weights(ann: *mut fann, min_weight: fann_type, max_weight: fann_type);

    /// Initialize the weights using Widrow + Nguyen's algorithm.
    ///
    /// This function behaves similarly to `fann_randomize_weights`. It will use the algorithm
    /// developed by Derrick Nguyen and Bernard Widrow to set the weights in such a way
    /// as to speed up training. This technique is not always successful, and in some cases can be
    /// less efficient than a purely random initialization.
    ///
    /// The algorithm requires access to the range of the input data (ie, largest and smallest
    /// input), and therefore accepts a second argument, `data`, which is the training data that
    /// will be used to train the network.
    ///
    /// # See also
    /// `fann_randomize_weights`, `fann_read_train_from_file`
    ///
    /// This function appears in FANN >= 1.1.0.
    pub fn fann_init_weights(ann: *mut fann, train_data: *mut fann_train_data);

    /// Prints the connections of the ANN in a compact matrix, for easy viewing of the internals
    /// of the ANN.
    ///
    /// The output from `fann_print_connections` on a small (2 2 1) network trained on the xor
    /// problem:
    ///
    /// ```text
    /// Layer / Neuron 012345
    /// L   1 / N    3 BBa...
    /// L   1 / N    4 BBA...
    /// L   1 / N    5 ......
    /// L   2 / N    6 ...BBA
    /// L   2 / N    7 ......
    /// ```
    ///
    /// This network has five real neurons and two bias neurons. This gives a total of seven
    /// neurons named from 0 to 6. The connections between these neurons can be seen in the matrix.
    /// "." is a place where there is no connection, while a character tells how strong the
    /// connection is on a scale from a-z. The two real neurons in the hidden layer (neuron 3 and 4
    /// in layer 1) have connections from the three neurons in the previous layer as is visible in
    /// the first two lines. The output neuron 6 has connections from the three neurons in the
    /// hidden layer 3 - 5 as is visible in the fourth line.
    ///
    /// To simplify the matrix output neurons are not visible as neurons that connections can come
    /// from, and input and bias neurons are not visible as neurons that connections can go to.
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_print_connections(ann: *mut fann);

    /// Prints all of the parameters and options of the ANN.
    ///
    /// This function appears in FANN >= 1.2.0.
    pub fn fann_print_parameters(ann: *mut fann);

    /// Get the number of input neurons.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_get_num_input(ann: *const fann) -> c_uint;

    /// Get the number of output neurons.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_get_num_output(ann: *const fann) -> c_uint;

    /// Get the total number of neurons in the entire network. This number does also include the
    /// bias neurons, so a 2-4-2 network has 2+4+2 +2(bias) = 10 neurons.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_get_total_neurons(ann: *const fann) -> c_uint;

    /// Get the total number of connections in the entire network.
    ///
    /// This function appears in FANN >= 1.0.0.
    pub fn fann_get_total_connections(ann: *const fann) -> c_uint;

    /// Get the type of neural network it was created as.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// # Returns
    /// The neural network type from enum `fann_network_type_enum`
    ///
    /// # See also
    /// `fann_network_type_enum`
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_get_network_type(ann: *const fann) -> fann_nettype_enum;

    /// Get the connection rate used when the network was created.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// # Returns
    /// The connection rate
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_get_connection_rate(ann: *const fann) -> c_float;

    /// Get the number of layers in the network.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// # Returns
    ///
    /// The number of layers in the neural network
    ///
    /// # Example
    ///
    /// ```
    /// // Obtain the number of layers in a neural network
    /// unsafe {
    ///     let ann = fann_sys::fann_create_standard(4, 2, 8, 9, 1);
    ///     assert_eq!(4, fann_sys::fann_get_num_layers(ann));
    /// }
    /// ```
    ///
    /// This function appears in FANN >= 2.1.0
    pub fn fann_get_num_layers(ann: *const fann) -> c_uint;

    /// Get the number of neurons in each layer in the network.
    ///
    /// Bias is not included so the layers match the `fann_create` functions.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// The layers array must be preallocated to accommodate at least `fann_num_layers` items.
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_get_layer_array(ann: *const fann, layers: *mut c_uint);

    /// Get the number of bias in each layer in the network.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// The bias array must be preallocated to accommodate at least `fann_num_layers` items.
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_get_bias_array(ann: *const fann, bias: *mut c_uint);

    /// Get the connections in the network.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// The connections array must be preallocated to accommodate at least
    /// `fann_get_total_connections` items.
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_get_connection_array(ann: *const fann, connections: *mut fann_connection);

    /// Set connections in the network.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// Only the weights can be changed, connections and weights are ignored
    /// if they do not already exist in the network.
    ///
    /// The array must accommodate `num_connections` items.
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_set_weight_array(ann: *mut fann,
                                 connections: *mut fann_connection,
                                 num_connections: c_uint);

    /// Set a connection in the network.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// Only the weights can be changed. The connection/weight is
    /// ignored if it does not already exist in the network.
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_set_weight(ann: *mut fann, from_neuron: c_uint,
                           to_neuron: c_uint, weight: fann_type);

    /// Store a pointer to user defined data. The pointer can be retrieved with `fann_get_user_data`
    /// for example in a callback. It is the user's responsibility to allocate and deallocate any
    /// data that the pointer might point to.
    ///
    /// # Parameters
    ///
    /// * `ann`       - A previously created neural network structure of type `fann` pointer.
    /// * `user_data` - A void pointer to user defined data.
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_set_user_data(ann: *mut fann, user_data: *mut c_void);

    /// Get a pointer to user defined data that was previously set with `fann_set_user_data`. It is
    /// the user's responsibility to allocate and deallocate any data that the pointer might point
    /// to.
    ///
    /// # Parameters
    ///
    /// * `ann` - A previously created neural network structure of type `fann` pointer.
    ///
    /// # Returns
    /// A void pointer to user defined data.
    ///
    /// This function appears in FANN >= 2.1.0.
    pub fn fann_get_user_data(ann: *mut fann) -> *mut c_void;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::fs::remove_file;
    use std::str::from_utf8;

    const EPSILON: f32 = 0.2;

    #[test]
    fn test_tutorial_example() {
        let c_trainfile = CString::new(&b"test_files/xor.data"[..]).unwrap();
        let p_trainfile = c_trainfile.as_ptr();
        let c_savefile = CString::new(&b"test_files/xor.net"[..]).unwrap();
        let p_savefile = c_savefile.as_ptr();
        // Train an ANN with a data set and then save the ANN to a file.
        let num_input = 2;
        let num_output = 1;
        let num_layers = 3;
        let num_neurons_hidden = 3;
        let desired_error = 0.001;
        let max_epochs = 500000;
        let epochs_between_reports = 1000;
        unsafe {
            let ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
            fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
            fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
            fann_train_on_file(ann, p_trainfile, max_epochs, epochs_between_reports, desired_error);
            fann_save(ann, p_savefile);
            fann_destroy(ann);
        }
        // Load the ANN and execute input.
        unsafe {
            let ann = fann_create_from_file(p_savefile);
            assert!(EPSILON > ( 1.0 - *fann_run(ann, [-1.0,  1.0].as_ptr())).abs());
            assert!(EPSILON > ( 1.0 - *fann_run(ann, [ 1.0, -1.0].as_ptr())).abs());
            assert!(EPSILON > (-1.0 - *fann_run(ann, [ 1.0,  1.0].as_ptr())).abs());
            assert!(EPSILON > (-1.0 - *fann_run(ann, [-1.0, -1.0].as_ptr())).abs());
            fann_destroy(ann);
        }
        // Delete the ANN file created by the test.
        remove_file(from_utf8(c_savefile.to_bytes()).unwrap()).unwrap();
    }
}
