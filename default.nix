{ nixpkgs ?
  builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/20.09.tar.gz"
}:
let
  pkgs = import nixpkgs {
    overlays = [
      (_: pkgs: {

        aml = let
          f = { stdenv, src, autoreconfHook, git, pkgconfig, numactl, hwloc }:
            stdenv.mkDerivation {
              src = ./.;
              name = "aml";
              nativeBuildInputs = [ autoreconfHook pkgconfig git ];
              buildInputs = [ hwloc numactl ];
            };
        in pkgs.lib.callPackageWith pkgs f { };

      })
    ];
  };

in pkgs
