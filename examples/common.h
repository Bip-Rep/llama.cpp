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

//
// Console utils
//

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_BOLD "\x1b[1m"

enum console_color_t
{
    CONSOLE_COLOR_DEFAULT = 0,
    CONSOLE_COLOR_PROMPT,
    CONSOLE_COLOR_USER_INPUT
};

struct console_state
{
    bool use_color = false;
    console_color_t color = CONSOLE_COLOR_DEFAULT;
};

void set_console_color(console_state &con_st, console_color_t color);

#if defined(_WIN32)
void win32_console_init(bool enable_color);
void win32_utf8_encode(const std::wstring &wstr, std::string &str);
#endif
