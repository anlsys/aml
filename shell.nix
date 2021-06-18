# development shell, includes aml dependencies and dev-related helpers
{ pkgs ? import ./. { } }:
with pkgs;
mkShell {
  inputsFrom = [ aml ];

  nativeBuildInputs = [ autoreconfHook pkgconfig ];

  buildInputs = [
    # deps for docs
    graphviz
    doxygen
    python3Packages.sphinx
    python3Packages.breathe
    python3Packages.sphinx_rtd_theme
    # deps for debug
    gdb
    valgrind
    # style checks
    clang-tools
    llvmPackages.clang-unwrapped.python
  ];

  CFLAGS =
    "-std=c99 -pedantic -Wall -Wextra -Werror -Wno-unused-but-set-parameter -Wno-builtin-declaration-mismatch";
}
