{ nixpkgs ?
  builtins.fetchTarball "https://github.com/NixOS/nixpkgs/archive/20.03.tar.gz"
}:
let
  pkgs = import nixpkgs {
    overlays = [
      (_: pkgs: {

        doxygen = pkgs.doxygen.overrideAttrs (old: rec {
          name = "doxygen-1.8.14";
          src = pkgs.fetchurl {
            urls = [
              "mirror://sourceforge/doxygen/${name}.src.tar.gz"
              "http://doxygen.nl/files/${name}.src.tar.gz"
            ];
            sha256 = "0XV+AnVe9vVv1F8fQ5hZi5IDgZSNb8+lj1ymqlb1nU0=";
          };

        });

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
