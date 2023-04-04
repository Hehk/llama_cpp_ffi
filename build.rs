extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

fn main() {
    generate_binding("llama");
    generate_binding("ggml");

    compile_cpp_files();
}

fn generate_binding(file: &str) {
    let bindings = bindgen::Builder::default()
        .header(format!("llama.cpp/{}.h", file))
        .clang_arg("-x")
        .clang_arg("c++")
        .ctypes_prefix("libc")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join(format!("{}.rs", file)))
        .expect("Couldn't write bindings!");
}

fn compile_cpp_files() {
    // Compile llama.cpp
    cc::Build::new()
        .cpp(true)
        .flag("-std=c++11")
        .file("llama.cpp/llama.cpp")
        .compile("libllama.a");

    // Compile ggml.c
    cc::Build::new()
        .file("llama.cpp/ggml.c")
        .compile("libggml.a");
}
