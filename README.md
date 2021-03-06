# fann-sys-rs

[![Build Status](https://travis-ci.org/afck/fann-sys-rs.svg?branch=master)](https://travis-ci.org/afck/fann-sys-rs)
[![Crates.io](http://meritbadge.herokuapp.com/fann-sys)](https://crates.io/crates/fann-sys)

Low-level [Rust](http://www.rust-lang.org/) bindings to the
[Fast Artificial Neural Networks](http://leenissen.dk/fann/wp/) library. The
[wrapper fann-rs](https://github.com/afck/fann-rs) provides a safe interface on
top of these.

[Documentation](https://afck.github.io/docs/fann-sys-rs/fann_sys)


## Usage

Add `fann-sys` and `libc` to the list of dependencies in your `Cargo.toml`:

```toml
[dependencies]
fann-sys = "*"
libc = "*"
```

and this to your crate root:

```rust
extern crate fann;
extern crate libc;
```
