#ifndef LLAMA_H
#define LLAMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string>
#include <vector>
#include <random>
#include <thread>

#ifdef LLAMA_SHARED
#ifdef _WIN32
#ifdef LLAMA_BUILD
#define LLAMA_API __declspec(dllexport)
#else
#define LLAMA_API __declspec(dllimport)
#endif
#else
#define LLAMA_API __attribute__((visibility("default")))
#endif
#else
#define LLAMA_API
#endif

#define LLAMA_FILE_VERSION 1
#define LLAMA_FILE_MAGIC 0x67676d66             // 'ggmf' in hex
#define LLAMA_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files

#ifdef __cplusplus
extern "C"
{
#endif

    struct gpt_params
    {
        int32_t seed = -1; // RNG seed
        int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
        int32_t n_predict = 128;    // new tokens to predict
        int32_t repeat_last_n = 64; // last n tokens to penalize
        int32_t n_parts = -1;       // amount of model parts (-1 = determine from model dimensions)
        int32_t n_ctx = 512;        // context size
        int32_t n_batch = 8;        // batch size for prompt processing
        int32_t n_keep = 0;         // number of tokens to keep from initial prompt

        // sampling parameters
        int32_t top_k = 40;
        float top_p = 0.95f;
        float temp = 0.80f;
        float repeat_penalty = 1.10f;

        std::string model = "models/lamma-7B/ggml-model.bin"; // model path
        std::string prompt = "";
        std::string input_prefix = ""; // string to prefix user inputs with

        std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

        bool memory_f16 = true;     // use f16 instead of f32 for memory kv
        bool random_prompt = false; // do not randomize prompt if none provided
        bool use_color = false;     // use color to distinguish generations and inputs
        bool interactive = false;   // interactive mode

        bool embedding = false;         // get only sentence embedding
        bool interactive_start = false; // wait for user input immediately

        bool instruct = false;       // instruction mode (used for Alpaca models)
        bool ignore_eos = false;     // do not stop generating after eos
        bool perplexity = false;     // compute perplexity over the prompt
        bool use_mlock = false;      // use mlock to keep model in memory
        bool mem_test = false;       // compute maximum memory usage
        bool verbose_prompt = false; // print prompt tokens before generation
    };

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_context;

    typedef int llama_token;

    typedef struct llama_token_data
    {
        llama_token id; // token id

        float p;    // probability of the token
        float plog; // log probability of the token

    } llama_token_data;

    typedef void (*llama_progress_callback)(double progress, void *ctx);

    struct llama_context_params
    {
        int n_ctx;   // text context
        int n_parts; // -1 for default
        int seed;    // RNG seed, 0 for random

        bool f16_kv;     // use fp16 for KV cache
        bool logits_all; // the llama_eval() call computes all logits, not just the last one
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mlock;  // force system to keep model in RAM
        bool embedding;  // embedding mode only

        // called with a progress value between 0 and 1, pass NULL to disable
        llama_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void *progress_callback_user_data;
    };

    LLAMA_API struct llama_context_params llama_context_default_params();

    // Various functions for loading a ggml llama model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    LLAMA_API struct llama_context *llama_init_from_file(
        const char *path_model,
        struct llama_context_params params);

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context *ctx);

    // TODO: not great API - very likely to change
    // Returns 0 on success
    LLAMA_API int llama_model_quantize(
        const char *fname_inp,
        const char *fname_out,
        int itype,
        int qk);

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    LLAMA_API int llama_eval(
        struct llama_context *ctx,
        const llama_token *tokens,
        int n_tokens,
        int n_past,
        int n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    LLAMA_API int llama_tokenize(
        struct llama_context *ctx,
        const char *text,
        llama_token *tokens,
        int n_max_tokens,
        bool add_bos);

    LLAMA_API int llama_n_vocab(struct llama_context *ctx);
    LLAMA_API int llama_n_ctx(struct llama_context *ctx);
    LLAMA_API int llama_n_embd(struct llama_context *ctx);

    // Token logits obtained from the last call to llama_eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    LLAMA_API float *llama_get_logits(struct llama_context *ctx);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    LLAMA_API float *llama_get_embeddings(struct llama_context *ctx);

    // Token Id -> String. Uses the vocabulary in the provided context
    LLAMA_API const char *llama_token_to_str(struct llama_context *ctx, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_token_bos();
    LLAMA_API llama_token llama_token_eos();

    // TODO: improve the last_n_tokens interface ?
    LLAMA_API llama_token llama_sample_top_p_top_k(
        struct llama_context *ctx,
        const llama_token *last_n_tokens_data,
        int last_n_tokens_size,
        int top_k,
        double top_p,
        double temp,
        double repeat_penalty);

    // Performance information
    LLAMA_API void llama_print_timings(struct llama_context *ctx);
    LLAMA_API void llama_reset_timings(struct llama_context *ctx);

    // Print system information
    LLAMA_API const char *llama_print_system_info(void);

#ifdef __cplusplus
}
#endif

#endif
