extern crate libc;

pub mod ffi;

use std::ffi::CString;

#[test]
fn test_tutorial_example() {
    unsafe {
        let num_input = 2;
        let num_output = 1;
        let num_layers = 3;
        let num_neurons_hidden = 3;
        let desired_error = 0.001;
        let max_epochs = 500000;
        let epochs_between_reports = 1000;
        let ann = ffi::fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

        ffi::fann_set_activation_function_hidden(
            ann, ffi::fann_activationfunc_enum::FANN_SIGMOID_SYMMETRIC);
        ffi::fann_set_activation_function_output(
            ann, ffi::fann_activationfunc_enum::FANN_SIGMOID_SYMMETRIC);

        let c_train_file = CString::new(&b"xor.data"[..]).unwrap();
        let c_save_file = CString::new(&b"xor_float.net"[..]).unwrap();
        ffi::fann_train_on_file(
            ann, c_train_file.as_ptr(), max_epochs, epochs_between_reports, desired_error);

        ffi::fann_save(ann, c_save_file.as_ptr());

        ffi::fann_destroy(ann);
    }
}
