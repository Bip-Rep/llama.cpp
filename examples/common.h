// Various helper functions and utilities

#pragma once

#include "llama.h"

//
// CLI argument parsing
//

bool gpt_params_parse(int argc, char **argv, gpt_params &params);

void gpt_print_usage(int argc, char **argv, const gpt_params &params);

std::string gpt_random_prompt(std::mt19937 &rng);

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(struct llama_context *ctx, const std::string &text, bool add_bos);
