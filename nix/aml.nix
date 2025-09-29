{ stdenv, autoreconfHook, git, pkgconf, numactl, hwloc, mpich}:
stdenv.mkDerivation {
  src = ../.;
  name = "aml";
  nativeBuildInputs = [ autoreconfHook pkgconf git ];
  buildInputs = [ hwloc numactl mpich ];
}
