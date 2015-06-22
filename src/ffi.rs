use libc::types::common::c95::FILE;
use libc::{c_char, c_float, c_int, c_uint, c_void};

/* Created with the help of rust-bindgen. */

pub type fann_type = c_float;

pub type Enum_fann_errno_enum = c_uint;
pub const FANN_E_NO_ERROR: c_uint = 0;
pub const FANN_E_CANT_OPEN_CONFIG_R: c_uint = 1;
pub const FANN_E_CANT_OPEN_CONFIG_W: c_uint = 2;
pub const FANN_E_WRONG_CONFIG_VERSION: c_uint = 3;
pub const FANN_E_CANT_READ_CONFIG: c_uint = 4;
pub const FANN_E_CANT_READ_NEURON: c_uint = 5;
pub const FANN_E_CANT_READ_CONNECTIONS: c_uint = 6;
pub const FANN_E_WRONG_NUM_CONNECTIONS: c_uint = 7;
pub const FANN_E_CANT_OPEN_TD_W: c_uint = 8;
pub const FANN_E_CANT_OPEN_TD_R: c_uint = 9;
pub const FANN_E_CANT_READ_TD: c_uint = 10;
pub const FANN_E_CANT_ALLOCATE_MEM: c_uint = 11;
pub const FANN_E_CANT_TRAIN_ACTIVATION: c_uint = 12;
pub const FANN_E_CANT_USE_ACTIVATION: c_uint = 13;
pub const FANN_E_TRAIN_DATA_MISMATCH: c_uint = 14;
pub const FANN_E_CANT_USE_TRAIN_ALG: c_uint = 15;
pub const FANN_E_TRAIN_DATA_SUBSET: c_uint = 16;
pub const FANN_E_INDEX_OUT_OF_BOUND: c_uint = 17;
pub const FANN_E_SCALE_NOT_PRESENT: c_uint = 18;

pub type Enum_fann_train_enum = c_uint;
pub const FANN_TRAIN_INCREMENTAL: c_uint = 0;
pub const FANN_TRAIN_BATCH: c_uint = 1;
pub const FANN_TRAIN_RPROP: c_uint = 2;
pub const FANN_TRAIN_QUICKPROP: c_uint = 3;

pub type Enum_fann_activationfunc_enum = c_uint;
pub const FANN_LINEAR: c_uint = 0;
pub const FANN_THRESHOLD: c_uint = 1;
pub const FANN_THRESHOLD_SYMMETRIC: c_uint = 2;
pub const FANN_SIGMOID: c_uint = 3;
pub const FANN_SIGMOID_STEPWISE: c_uint = 4;
pub const FANN_SIGMOID_SYMMETRIC: c_uint = 5;
pub const FANN_SIGMOID_SYMMETRIC_STEPWISE: c_uint = 6;
pub const FANN_GAUSSIAN: c_uint = 7;
pub const FANN_GAUSSIAN_SYMMETRIC: c_uint = 8;
pub const FANN_GAUSSIAN_STEPWISE: c_uint = 9;
pub const FANN_ELLIOT: c_uint = 10;
pub const FANN_ELLIOT_SYMMETRIC: c_uint = 11;
pub const FANN_LINEAR_PIECE: c_uint = 12;
pub const FANN_LINEAR_PIECE_SYMMETRIC: c_uint = 13;
pub const FANN_SIN_SYMMETRIC: c_uint = 14;
pub const FANN_COS_SYMMETRIC: c_uint = 15;
pub const FANN_SIN: c_uint = 16;
pub const FANN_COS: c_uint = 17;

pub type Enum_fann_errorfunc_enum = c_uint;
pub const FANN_ERRORFUNC_LINEAR: c_uint = 0;
pub const FANN_ERRORFUNC_TANH: c_uint = 1;

pub type Enum_fann_stopfunc_enum = c_uint;
pub const FANN_STOPFUNC_MSE: c_uint = 0;
pub const FANN_STOPFUNC_BIT: c_uint = 1;

pub type Enum_fann_nettype_enum = c_uint;
pub const FANN_NETTYPE_LAYER: c_uint = 0;
pub const FANN_NETTYPE_SHORTCUT: c_uint = 1;

pub type fann_callback_type = Option<
    extern "C" fn(ann: *mut Struct_fann,
                  train: *mut Struct_fann_train_data,
                  max_epochs: c_uint,
                  epochs_between_reports: c_uint,
                  desired_error: c_float,
                  epochs: c_uint) -> c_int>;

#[repr(C)]
#[derive(Copy)]
pub struct Struct_fann_neuron {
    pub first_con: c_uint,
    pub last_con: c_uint,
    pub sum: fann_type,
    pub value: fann_type,
    pub activation_steepness: fann_type,
    pub activation_function: Enum_fann_activationfunc_enum,
}

impl ::std::clone::Clone for Struct_fann_neuron {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_fann_neuron {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_fann_layer {
    pub first_neuron: *mut Struct_fann_neuron,
    pub last_neuron: *mut Struct_fann_neuron,
}

impl ::std::clone::Clone for Struct_fann_layer {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_fann_layer {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_fann_error {
    pub errno_f: Enum_fann_errno_enum,
    pub error_log: *mut FILE,
    pub errstr: *mut c_char,
}

impl ::std::clone::Clone for Struct_fann_error {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_fann_error {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_fann {
    pub errno_f: Enum_fann_errno_enum,
    pub error_log: *mut FILE,
    pub errstr: *mut c_char,
    pub learning_rate: c_float,
    pub learning_momentum: c_float,
    pub connection_rate: c_float,
    pub network_type: Enum_fann_nettype_enum,
    pub first_layer: *mut Struct_fann_layer,
    pub last_layer: *mut Struct_fann_layer,
    pub total_neurons: c_uint,
    pub num_input: c_uint,
    pub num_output: c_uint,
    pub weights: *mut fann_type,
    pub connections: *mut *mut Struct_fann_neuron,
    pub train_errors: *mut fann_type,
    pub training_algorithm: Enum_fann_train_enum,
    pub total_connections: c_uint,
    pub output: *mut fann_type,
    pub num_MSE: c_uint,
    pub MSE_value: c_float,
    pub num_bit_fail: c_uint,
    pub bit_fail_limit: fann_type,
    pub train_error_function: Enum_fann_errorfunc_enum,
    pub train_stop_function: Enum_fann_stopfunc_enum,
    pub callback: fann_callback_type,
    pub user_data: *mut c_void,
    pub cascade_output_change_fraction: c_float,
    pub cascade_output_stagnation_epochs: c_uint,
    pub cascade_candidate_change_fraction: c_float,
    pub cascade_candidate_stagnation_epochs: c_uint,
    pub cascade_best_candidate: c_uint,
    pub cascade_candidate_limit: fann_type,
    pub cascade_weight_multiplier: fann_type,
    pub cascade_max_out_epochs: c_uint,
    pub cascade_max_cand_epochs: c_uint,
    pub cascade_activation_functions: *mut Enum_fann_activationfunc_enum,
    pub cascade_activation_functions_count: c_uint,
    pub cascade_activation_steepnesses: *mut fann_type,
    pub cascade_activation_steepnesses_count: c_uint,
    pub cascade_num_candidate_groups: c_uint,
    pub cascade_candidate_scores: *mut fann_type,
    pub total_neurons_allocated: c_uint,
    pub total_connections_allocated: c_uint,
    pub quickprop_decay: c_float,
    pub quickprop_mu: c_float,
    pub rprop_increase_factor: c_float,
    pub rprop_decrease_factor: c_float,
    pub rprop_delta_min: c_float,
    pub rprop_delta_max: c_float,
    pub rprop_delta_zero: c_float,
    pub train_slopes: *mut fann_type,
    pub prev_steps: *mut fann_type,
    pub prev_train_slopes: *mut fann_type,
    pub prev_weights_deltas: *mut fann_type,
    pub scale_mean_in: *mut c_float,
    pub scale_deviation_in: *mut c_float,
    pub scale_new_min_in: *mut c_float,
    pub scale_factor_in: *mut c_float,
    pub scale_mean_out: *mut c_float,
    pub scale_deviation_out: *mut c_float,
    pub scale_new_min_out: *mut c_float,
    pub scale_factor_out: *mut c_float,
}

impl ::std::clone::Clone for Struct_fann {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_fann {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_fann_connection {
    pub from_neuron: c_uint,
    pub to_neuron: c_uint,
    pub weight: fann_type,
}

impl ::std::clone::Clone for Struct_fann_connection {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_fann_connection {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

#[repr(C)]
#[derive(Copy)]
pub struct Struct_fann_train_data {
    pub errno_f: Enum_fann_errno_enum,
    pub error_log: *mut FILE,
    pub errstr: *mut c_char,
    pub num_data: c_uint,
    pub num_input: c_uint,
    pub num_output: c_uint,
    pub input: *mut *mut fann_type,
    pub output: *mut *mut fann_type,
}

impl ::std::clone::Clone for Struct_fann_train_data {
    fn clone(&self) -> Self { *self }
}

impl ::std::default::Default for Struct_fann_train_data {
    fn default() -> Self { unsafe { ::std::mem::zeroed() } }
}

#[link(name = "fann")]
extern "C" {
    pub static mut fann_default_error_log: *mut FILE;

    pub fn fann_set_error_log(errdat: *mut Struct_fann_error, log_file: *mut FILE);
    pub fn fann_get_errno(errdat: *mut Struct_fann_error) -> Enum_fann_errno_enum;
    pub fn fann_reset_errno(errdat: *mut Struct_fann_error);
    pub fn fann_reset_errstr(errdat: *mut Struct_fann_error);
    pub fn fann_get_errstr(errdat: *mut Struct_fann_error) -> *mut c_char;
    pub fn fann_print_error(errdat: *mut Struct_fann_error);
    pub fn fann_allocate_structure(num_layers: c_uint) -> *mut Struct_fann;
    pub fn fann_allocate_neurons(ann: *mut Struct_fann);
    pub fn fann_allocate_connections(ann: *mut Struct_fann);
    pub fn fann_save_internal(ann: *mut Struct_fann,
                              configuration_file: *const c_char,
                              save_as_fixed: c_uint) -> c_int;
    pub fn fann_save_internal_fd(ann: *mut Struct_fann, conf: *mut FILE,
                                 configuration_file: *const c_char,
                                 save_as_fixed: c_uint) -> c_int;
    pub fn fann_save_train_internal(data: *mut Struct_fann_train_data,
                                    filename: *const c_char,
                                    save_as_fixed: c_uint,
                                    decimal_point: c_uint) -> c_int;
    pub fn fann_save_train_internal_fd(data: *mut Struct_fann_train_data,
                                       file: *mut FILE,
                                       filename: *const c_char,
                                       save_as_fixed: c_uint,
                                       decimal_point: c_uint) -> c_int;
    pub fn fann_update_stepwise(ann: *mut Struct_fann);
    pub fn fann_seed_rand();
    pub fn fann_error(errdat: *mut Struct_fann_error, errno_f: Enum_fann_errno_enum, ...);
    pub fn fann_init_error_data(errdat: *mut Struct_fann_error);
    pub fn fann_create_from_fd(conf: *mut FILE, configuration_file: *const c_char)
        -> *mut Struct_fann;
    pub fn fann_read_train_from_fd(file: *mut FILE, filename: *const c_char)
        -> *mut Struct_fann_train_data;
    pub fn fann_compute_MSE(ann: *mut Struct_fann, desired_output: *mut fann_type);
    pub fn fann_update_output_weights(ann: *mut Struct_fann);
    pub fn fann_backpropagate_MSE(ann: *mut Struct_fann);
    pub fn fann_update_weights(ann: *mut Struct_fann);
    pub fn fann_update_slopes_batch(ann: *mut Struct_fann,
                                    layer_begin: *mut Struct_fann_layer,
                                    layer_end: *mut Struct_fann_layer);
    pub fn fann_update_weights_quickprop(ann: *mut Struct_fann,
                                         num_data: c_uint,
                                         first_weight: c_uint,
                                         past_end: c_uint);
    pub fn fann_update_weights_batch(ann: *mut Struct_fann,
                                     num_data: c_uint,
                                     first_weight: c_uint,
                                     past_end: c_uint);
    pub fn fann_update_weights_irpropm(ann: *mut Struct_fann,
                                       first_weight: c_uint,
                                       past_end: c_uint);
    pub fn fann_clear_train_arrays(ann: *mut Struct_fann);
    pub fn fann_activation(ann: *mut Struct_fann,
                           activation_function: c_uint,
                           steepness: fann_type, value: fann_type) -> fann_type;
    pub fn fann_activation_derived(activation_function: c_uint,
                                   steepness: fann_type, value: fann_type,
                                   sum: fann_type) -> fann_type;
    pub fn fann_desired_error_reached(ann: *mut Struct_fann,
                                      desired_error: c_float) -> c_int;
    pub fn fann_train_outputs(ann: *mut Struct_fann,
                              data: *mut Struct_fann_train_data,
                              desired_error: c_float) -> c_int;
    pub fn fann_train_outputs_epoch(ann: *mut Struct_fann,
                                    data: *mut Struct_fann_train_data) -> c_float;
    pub fn fann_train_candidates(ann: *mut Struct_fann,
                                 data: *mut Struct_fann_train_data) -> c_int;
    pub fn fann_train_candidates_epoch(ann: *mut Struct_fann,
                                       data: *mut Struct_fann_train_data) -> fann_type;
    pub fn fann_install_candidate(ann: *mut Struct_fann);
    pub fn fann_initialize_candidates(ann: *mut Struct_fann) -> c_int;
    pub fn fann_set_shortcut_connections(ann: *mut Struct_fann);
    pub fn fann_allocate_scale(ann: *mut Struct_fann) -> c_int;
    pub fn fann_train(ann: *mut Struct_fann, input: *mut fann_type,
                      desired_output: *mut fann_type);
    pub fn fann_test(ann: *mut Struct_fann, input: *mut fann_type,
                     desired_output: *mut fann_type) -> *mut fann_type;
    pub fn fann_get_MSE(ann: *mut Struct_fann) -> c_float;
    pub fn fann_get_bit_fail(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_reset_MSE(ann: *mut Struct_fann);
    pub fn fann_train_on_data(ann: *mut Struct_fann,
                              data: *mut Struct_fann_train_data,
                              max_epochs: c_uint,
                              epochs_between_reports: c_uint,
                              desired_error: c_float);
    pub fn fann_train_on_file(ann: *mut Struct_fann,
                              filename: *const c_char,
                              max_epochs: c_uint,
                              epochs_between_reports: c_uint,
                              desired_error: c_float);
    pub fn fann_train_epoch(ann: *mut Struct_fann, data: *mut Struct_fann_train_data) -> c_float;
    pub fn fann_test_data(ann: *mut Struct_fann, data: *mut Struct_fann_train_data) -> c_float;
    pub fn fann_read_train_from_file(filename: *const c_char) -> *mut Struct_fann_train_data;
    pub fn fann_create_train_from_callback(num_data: c_uint, num_input: c_uint, num_output: c_uint,
        user_function: Option<extern "C" fn(arg1: c_uint, arg2: c_uint, arg3: c_uint,
                                            arg4: *mut fann_type, arg5: *mut fann_type)>)
        -> *mut Struct_fann_train_data;
    pub fn fann_destroy_train(train_data: *mut Struct_fann_train_data);
    pub fn fann_shuffle_train_data(train_data: *mut Struct_fann_train_data);
    pub fn fann_scale_train(ann: *mut Struct_fann, data: *mut Struct_fann_train_data);
    pub fn fann_descale_train(ann: *mut Struct_fann, data: *mut Struct_fann_train_data);
    pub fn fann_set_input_scaling_params(ann: *mut Struct_fann,
                                         data: *const Struct_fann_train_data,
                                         new_input_min: c_float,
                                         new_input_max: c_float) -> c_int;
    pub fn fann_set_output_scaling_params(ann: *mut Struct_fann,
                                          data: *const Struct_fann_train_data,
                                          new_output_min: c_float,
                                          new_output_max: c_float) -> c_int;
    pub fn fann_set_scaling_params(ann: *mut Struct_fann,
                                   data: *const Struct_fann_train_data,
                                   new_input_min: c_float,
                                   new_input_max: c_float,
                                   new_output_min: c_float,
                                   new_output_max: c_float) -> c_int;
    pub fn fann_clear_scaling_params(ann: *mut Struct_fann) -> c_int;
    pub fn fann_scale_input(ann: *mut Struct_fann, input_vector: *mut fann_type);
    pub fn fann_scale_output(ann: *mut Struct_fann, output_vector: *mut fann_type);
    pub fn fann_descale_input(ann: *mut Struct_fann, input_vector: *mut fann_type);
    pub fn fann_descale_output(ann: *mut Struct_fann, output_vector: *mut fann_type);
    pub fn fann_scale_input_train_data(train_data: *mut Struct_fann_train_data,
                                       new_min: fann_type, new_max: fann_type);
    pub fn fann_scale_output_train_data(train_data: *mut Struct_fann_train_data,
                                        new_min: fann_type, new_max: fann_type);
    pub fn fann_scale_train_data(train_data: *mut Struct_fann_train_data,
                                 new_min: fann_type, new_max: fann_type);
    pub fn fann_merge_train_data(data1: *mut Struct_fann_train_data,
                                 data2: *mut Struct_fann_train_data) -> *mut Struct_fann_train_data;
    pub fn fann_duplicate_train_data(data: *mut Struct_fann_train_data)
        -> *mut Struct_fann_train_data;
    pub fn fann_subset_train_data(data: *mut Struct_fann_train_data, pos: c_uint, length: c_uint)
        -> *mut Struct_fann_train_data;
    pub fn fann_length_train_data(data: *mut Struct_fann_train_data) -> c_uint;
    pub fn fann_num_input_train_data(data: *mut Struct_fann_train_data) -> c_uint;
    pub fn fann_num_output_train_data(data: *mut Struct_fann_train_data) -> c_uint;
    pub fn fann_save_train(data: *mut Struct_fann_train_data, filename: *const c_char) -> c_int;
    pub fn fann_save_train_to_fixed(data: *mut Struct_fann_train_data,
                                    filename: *const c_char,
                                    decimal_point: c_uint) -> c_int;
    pub fn fann_get_training_algorithm(ann: *mut Struct_fann) -> Enum_fann_train_enum;
    pub fn fann_set_training_algorithm(ann: *mut Struct_fann,
                                       training_algorithm: Enum_fann_train_enum);
    pub fn fann_get_learning_rate(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_learning_rate(ann: *mut Struct_fann, learning_rate: c_float);
    pub fn fann_get_learning_momentum(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_learning_momentum(ann: *mut Struct_fann, learning_momentum: c_float);
    pub fn fann_get_activation_function(ann: *mut Struct_fann, layer: c_int, neuron: c_int)
        -> Enum_fann_activationfunc_enum;
    pub fn fann_set_activation_function(ann: *mut Struct_fann,
                                        activation_function: Enum_fann_activationfunc_enum,
                                        layer: c_int,
                                        neuron: c_int);
    pub fn fann_set_activation_function_layer(ann: *mut Struct_fann,
                                              activation_function: Enum_fann_activationfunc_enum,
                                              layer: c_int);
    pub fn fann_set_activation_function_hidden(ann: *mut Struct_fann,
                                               activation_function: Enum_fann_activationfunc_enum);
    pub fn fann_set_activation_function_output(ann: *mut Struct_fann,
                                               activation_function: Enum_fann_activationfunc_enum);
    pub fn fann_get_activation_steepness(ann: *mut Struct_fann, layer: c_int, neuron: c_int)
        -> fann_type;
    pub fn fann_set_activation_steepness(ann: *mut Struct_fann,
                                         steepness: fann_type,
                                         layer: c_int,
                                         neuron: c_int);
    pub fn fann_set_activation_steepness_layer(ann: *mut Struct_fann,
                                               steepness: fann_type,
                                               layer: c_int);
    pub fn fann_set_activation_steepness_hidden(ann: *mut Struct_fann,
                                                steepness: fann_type);
    pub fn fann_set_activation_steepness_output(ann: *mut Struct_fann,
                                                steepness: fann_type);
    pub fn fann_get_train_error_function(ann: *mut Struct_fann) -> Enum_fann_errorfunc_enum;
    pub fn fann_set_train_error_function(ann: *mut Struct_fann,
                                         train_error_function: Enum_fann_errorfunc_enum);
    pub fn fann_get_train_stop_function(ann: *mut Struct_fann) -> Enum_fann_stopfunc_enum;
    pub fn fann_set_train_stop_function(ann: *mut Struct_fann,
                                        train_stop_function: Enum_fann_stopfunc_enum);
    pub fn fann_get_bit_fail_limit(ann: *mut Struct_fann) -> fann_type;
    pub fn fann_set_bit_fail_limit(ann: *mut Struct_fann, bit_fail_limit: fann_type);
    pub fn fann_set_callback(ann: *mut Struct_fann, callback: fann_callback_type);
    pub fn fann_get_quickprop_decay(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_quickprop_decay(ann: *mut Struct_fann, quickprop_decay: c_float);
    pub fn fann_get_quickprop_mu(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_quickprop_mu(ann: *mut Struct_fann, quickprop_mu: c_float);
    pub fn fann_get_rprop_increase_factor(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_rprop_increase_factor(ann: *mut Struct_fann, rprop_increase_factor: c_float);
    pub fn fann_get_rprop_decrease_factor(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_rprop_decrease_factor(ann: *mut Struct_fann, rprop_decrease_factor: c_float);
    pub fn fann_get_rprop_delta_min(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_rprop_delta_min(ann: *mut Struct_fann, rprop_delta_min: c_float);
    pub fn fann_get_rprop_delta_max(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_rprop_delta_max(ann: *mut Struct_fann, rprop_delta_max: c_float);
    pub fn fann_get_rprop_delta_zero(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_rprop_delta_zero(ann: *mut Struct_fann, rprop_delta_max: c_float);
    pub fn fann_cascadetrain_on_data(ann: *mut Struct_fann,
                                     data: *mut Struct_fann_train_data,
                                     max_neurons: c_uint,
                                     neurons_between_reports: c_uint,
                                     desired_error: c_float);
    pub fn fann_cascadetrain_on_file(ann: *mut Struct_fann,
                                     filename: *const c_char,
                                     max_neurons: c_uint,
                                     neurons_between_reports: c_uint,
                                     desired_error: c_float);
    pub fn fann_get_cascade_output_change_fraction(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_cascade_output_change_fraction(ann: *mut Struct_fann,
                                                   cascade_output_change_fraction: c_float);
    pub fn fann_get_cascade_output_stagnation_epochs(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_set_cascade_output_stagnation_epochs(ann: *mut Struct_fann,
                                                     cascade_output_stagnation_epochs: c_uint)
    ;
    pub fn fann_get_cascade_candidate_change_fraction(ann: *mut Struct_fann) -> c_float;
    pub fn fann_set_cascade_candidate_change_fraction(ann: *mut Struct_fann,
                                                      cascade_candidate_change_fraction: c_float)
    ;
    pub fn fann_get_cascade_candidate_stagnation_epochs(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_set_cascade_candidate_stagnation_epochs(ann: *mut Struct_fann,
        cascade_candidate_stagnation_epochs: c_uint);
    pub fn fann_get_cascade_weight_multiplier(ann: *mut Struct_fann) -> fann_type;
    pub fn fann_set_cascade_weight_multiplier(ann: *mut Struct_fann,
                                              cascade_weight_multiplier: fann_type);
    pub fn fann_get_cascade_candidate_limit(ann: *mut Struct_fann) -> fann_type;
    pub fn fann_set_cascade_candidate_limit(ann: *mut Struct_fann,
                                            cascade_candidate_limit: fann_type);
    pub fn fann_get_cascade_max_out_epochs(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_set_cascade_max_out_epochs(ann: *mut Struct_fann,
                                           cascade_max_out_epochs: c_uint);
    pub fn fann_get_cascade_max_cand_epochs(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_set_cascade_max_cand_epochs(ann: *mut Struct_fann,
                                            cascade_max_cand_epochs: c_uint);
    pub fn fann_get_cascade_num_candidates(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_cascade_activation_functions_count(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_cascade_activation_functions(ann: *mut Struct_fann)
        -> *mut Enum_fann_activationfunc_enum;
    pub fn fann_set_cascade_activation_functions(ann: *mut Struct_fann,
        cascade_activation_functions: *mut Enum_fann_activationfunc_enum,                                                 cascade_activation_functions_count: c_uint);
    pub fn fann_get_cascade_activation_steepnesses_count(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_cascade_activation_steepnesses(ann: *mut Struct_fann) -> *mut fann_type;
    pub fn fann_set_cascade_activation_steepnesses(ann: *mut Struct_fann,
        cascade_activation_steepnesses: *mut fann_type,
        cascade_activation_steepnesses_count: c_uint);
    pub fn fann_get_cascade_num_candidate_groups(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_set_cascade_num_candidate_groups(ann: *mut Struct_fann,
                                                 cascade_num_candidate_groups: c_uint);
    pub fn fann_create_from_file(configuration_file: *const c_char) -> *mut Struct_fann;
    pub fn fann_save(ann: *mut Struct_fann, configuration_file: *const c_char) -> c_int;
    pub fn fann_save_to_fixed(ann: *mut Struct_fann, configuration_file: *const c_char) -> c_int;
    pub fn fann_create_standard(num_layers: c_uint, ...) -> *mut Struct_fann;
    pub fn fann_create_standard_array(num_layers: c_uint, layers: *const c_uint)
        -> *mut Struct_fann;
    pub fn fann_create_sparse(connection_rate: c_float, num_layers: c_uint, ...)
        -> *mut Struct_fann;
    pub fn fann_create_sparse_array(connection_rate: c_float,
                                    num_layers: c_uint,
                                    layers: *const c_uint) -> *mut Struct_fann;
    pub fn fann_create_shortcut(num_layers: c_uint, ...) -> *mut Struct_fann;
    pub fn fann_create_shortcut_array(num_layers: c_uint, layers: *const c_uint)
        -> *mut Struct_fann;
    pub fn fann_destroy(ann: *mut Struct_fann);
    pub fn fann_run(ann: *mut Struct_fann, input: *mut fann_type) -> *mut fann_type;
    pub fn fann_randomize_weights(ann: *mut Struct_fann,
                                  min_weight: fann_type,
                                  max_weight: fann_type);
    pub fn fann_init_weights(ann: *mut Struct_fann,
                             train_data: *mut Struct_fann_train_data);
    pub fn fann_print_connections(ann: *mut Struct_fann);
    pub fn fann_print_parameters(ann: *mut Struct_fann);
    pub fn fann_get_num_input(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_num_output(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_total_neurons(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_total_connections(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_network_type(ann: *mut Struct_fann) -> Enum_fann_nettype_enum;
    pub fn fann_get_connection_rate(ann: *mut Struct_fann) -> c_float;
    pub fn fann_get_num_layers(ann: *mut Struct_fann) -> c_uint;
    pub fn fann_get_layer_array(ann: *mut Struct_fann, layers: *mut c_uint);
    pub fn fann_get_bias_array(ann: *mut Struct_fann, bias: *mut c_uint);
    pub fn fann_get_connection_array(ann: *mut Struct_fann,
                                     connections: *mut Struct_fann_connection);
    pub fn fann_set_weight_array(ann: *mut Struct_fann,
                                 connections: *mut Struct_fann_connection,
                                 num_connections: c_uint);
    pub fn fann_set_weight(ann: *mut Struct_fann, from_neuron: c_uint,
                           to_neuron: c_uint, weight: fann_type);
    pub fn fann_set_user_data(ann: *mut Struct_fann, user_data: *mut c_void);
    pub fn fann_get_user_data(ann: *mut Struct_fann) -> *mut c_void;
}
