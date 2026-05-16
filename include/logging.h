#pragma once

#define ANSI_RESET  "\033[0m"
#define ANSI_BOLD   "\033[1m"
#define ANSI_DIM    "\033[2m"
#define ANSI_SEND   "\033[38;5;214m"  // orange  – sender
#define ANSI_RECV   "\033[38;5;75m"   // blue    – receiver
#define ANSI_BOTH   "\033[38;5;183m"  // purple  – both (shouldn't happen in tree)
#define ANSI_IDLE   "\033[38;5;240m"  // dark gray – idle
