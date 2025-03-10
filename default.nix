{ pkgs ? import (builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/24.05.tar.gz") {}
}:
pkgs // rec {
  stdenv = pkgs.gcc12Stdenv;
  aml = pkgs.callPackage ./nix/aml.nix { inherit stdenv; };
}
