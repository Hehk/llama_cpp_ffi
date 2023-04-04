#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod ggml;

use std::ffi::CString;
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/llama.rs"));

impl llama_context_params {
    fn new() -> llama_context_params {
        unsafe { llama_context_default_params() }
    }
}

pub struct LlamaContext {
    ctx: *mut llama_context,
}

impl LlamaContext {
    pub fn new_from_file<P: AsRef<Path>>(
        path: P,
        params: llama_context_params,
    ) -> Result<Self, &'static str> {
        let path_model = CString::new(path.as_ref().to_string_lossy().into_owned())
            .map_err(|_| "Failed to convert path to CString")?;

        let ctx = unsafe { llama_init_from_file(path_model.as_ptr(), params) };
        if ctx.is_null() {
            Err("Failed to initialize Llama context from file")
        } else {
            Ok(Self { ctx })
        }
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) }
    }
}

mod tests {
    use std::{ffi::CStr, io::Write, path};

    use super::*;

    // I am just stepping through the code in main.cpp and writing tests for each step
    // some of these may be excessive but I am just trying to get a feel for how to use the library

    #[test]
    fn param_creation_works() {
        let params = llama_context_params::new();
        print!("{:?}", params);
        assert_eq!(params.n_ctx, 512);
    }

    #[test]
    fn load_file_works() {
        let params = llama_context_params::new();
        let path_model = "models/gpt4all-lora-quantized.bin";

        let ctx = LlamaContext::new_from_file(path_model, params)
            .expect("Failed to load Llama context from file");

        assert!(!ctx.ctx.is_null());
    }

    #[test]
    fn tokenizer_works() {
        let params = llama_context_params::new();
        let path_model = "models/gpt4all-lora-quantized.bin";

        let mut ctx = LlamaContext::new_from_file(path_model, params)
            .expect("Failed to load Llama context from file");

        let input_text = "This is a test.";
        let add_bos = true;

        let c_input_text =
            CString::new(input_text).expect("Failed to convert input_text to CString");

        let max_tokens = (input_text.len() + add_bos as usize) as i32;
        let mut tokens: Vec<llama_token> = vec![llama_token::default(); max_tokens as usize];

        let n_tokens = unsafe {
            llama_tokenize(
                ctx.ctx,
                c_input_text.as_ptr(),
                tokens.as_mut_ptr(),
                max_tokens,
                add_bos,
            )
        };

        println!("len:{:?} tokens:{:?}", n_tokens, tokens);
        assert!(n_tokens >= 0, "Tokenization failed");
        println!("len:{:?} tokens:{:?}", n_tokens, tokens);
        tokens.resize(n_tokens as usize, 0);

        assert!(!tokens.is_empty(), "Tokenized output is empty");
    }

    #[test]
    fn generate_response() {
        let params = llama_context_params::new();
        let path_model = "models/gpt4all-lora-quantized.bin";

        let mut ctx = LlamaContext::new_from_file(path_model, params)
            .expect("Failed to load Llama context from file");

        let input_text = "This is a test.";
        let add_bos = true;

        let c_input_text =
            CString::new(input_text).expect("Failed to convert input_text to CString");

        let max_tokens = input_text.len() + add_bos as usize;
        let mut embd_inp: Vec<i32> = vec![llama_token::default(); max_tokens];

        let n_tokens = unsafe {
            llama_tokenize(
                ctx.ctx,
                c_input_text.as_ptr(),
                embd_inp.as_mut_ptr(),
                max_tokens.try_into().unwrap(),
                add_bos,
            )
        };
        embd_inp.resize(n_tokens as usize, 0);

        let n_predict = 128;
        let n_threads = 4;
        let mut n_remain = n_predict;
        // TODO get this from the model
        let n_ctx = 512;
        let mut n_past = 0;

        let mut embd: Vec<i32> = Vec::new();
        let mut n_consumed = 0;
        let mut last_n_tokens: Vec<llama_token> = vec![0; n_ctx];

        println!("embd len: {:?}", embd.len());
        println!("embd content: {:?}", embd);
        println!("embd_inp len: {:?}", embd_inp.len());

        while n_remain != 0 {
            if embd.len() > 0 {
                if n_past + embd.len() > n_ctx {
                    let n_keep = 0;
                    let n_left = n_past - n_keep;

                    n_past = n_keep;

                    // insert n_left/2 tokens at the start of embd from last_n_tokens
                    embd.splice(0..0, last_n_tokens.iter().cloned().take(n_left / 2));
                }

                unsafe {
                    if llama_eval(
                        ctx.ctx,
                        embd.as_ptr(),
                        embd.len().try_into().unwrap(),
                        n_past.try_into().unwrap(),
                        n_threads,
                    ) != 0
                    {
                        panic!("Failed to eval");
                    }
                }
            }

            n_past += embd.len();
            embd.clear();

            if embd.len() <= n_consumed {
                let mut id: llama_token = 0;
                let top_k = 40;
                let top_p = 0.9;
                let temp = 0.8;
                let repeat_penalty = 1.1;
                let repeat_last_n = 64;

                print!("{} ", id);
                let logits = unsafe { llama_get_logits(ctx.ctx) };
                let id = unsafe {
                    llama_sample_top_p_top_k(
                        ctx.ctx,
                        last_n_tokens
                            .as_ptr()
                            .offset((n_ctx - repeat_last_n) as isize),
                        repeat_last_n as i32,
                        top_k,
                        top_p,
                        temp,
                        repeat_penalty,
                    )
                };

                embd.push(id);
                n_remain -= 1;
            } else {
                while embd_inp.len() > n_consumed {
                    embd.push(embd_inp[n_consumed]);
                    last_n_tokens.remove(0);
                    last_n_tokens.push(embd_inp[n_consumed]);
                    n_consumed += 1;
                    if embd.len() >= 8 {
                        break;
                    }
                }
            }

            let output = unsafe {
                // loop through embd_inp and get a string for each
                embd.iter()
                    .map(|token| {
                        let cstr_ptr = llama_token_to_str(ctx.ctx, *token);
                        let cstr = CStr::from_ptr(cstr_ptr);
                        cstr.to_string_lossy().into_owned()
                    })
                    .collect::<Vec<String>>()
                    .join("")
            };
            print!("{}", output);
        }
    }
}
