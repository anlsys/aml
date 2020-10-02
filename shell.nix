# development shell, includes aml dependencies and dev-related helpers
# defined by argopkgs nix pkg record
{ pkgs ? import (
  builtins.fetchGit {
    url = "https://xgitlab.cels.anl.gov/argo/argopkgs.git";
    ref = "master";
    rev = "56f18ba59462dd7e073417bef9ff15c739905c8c";
  }) {} }:
with pkgs;
pkgs.mkShell {
	name = "aml";
	nativeBuildInputs = [ autoreconfHook pkgconfig ];
        buildInputs = [
          # dependencies for the code
          hwloc
          numactl
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
          (clang-tools.override (o:{llvmPackages=pkgs.llvmPackages_7;}))
          llvmPackages_7.clang-unwrapped.python
        ];
        CFLAGS = "-std=c99 -pedantic -Wall -Wextra -Werror -Wno-unused-but-set-parameter -Wno-builtin-declaration-mismatch";
}
