{ stdenv, autoreconfHook, git, pkgconf, numactl, hwloc}:
stdenv.mkDerivation {
  src = ../.;
  name = "aml";
  nativeBuildInputs = [ autoreconfHook pkgconf git ];
  buildInputs = [ hwloc numactl ];
}
