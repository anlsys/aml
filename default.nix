{ nixpkgs ?
  builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/20.03.tar.gz"
}:
let
  pkgs = import nixpkgs {
    overlays = [
      (_: pkgs: {

        hwloc = pkgs.hwloc.overrideAttrs (old: {
          name = "hwloc-2";
          src = pkgs.fetchurl {
            url =
              "https://download.open-mpi.org/release/hwloc/v2.1/hwloc-2.1.0.tar.gz";
            sha256 = "0mdqa9w1p6cmli6976v4wi0sw9r4p5prkj7lzfd1877wk11c9c73";
          };
        });

        aml = let
          f = { stdenv, src, autoreconfHook, pkgconfig, numactl, hwloc }:
            stdenv.mkDerivation {
              src = ./.;
              name = "aml";
              nativeBuildInputs = [ autoreconfHook pkgconfig ];
              buildInputs = [ hwloc numactl ];
            };
        in pkgs.lib.callPackageWith pkgs f { };

      })
    ];
  };

in pkgs
