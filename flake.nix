{
  description = "Development shell for CUDA-backed PyTorch and Numba.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Adjust if needed
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
      };
      cudatookit-with-cudart-to-lib64 = pkgs.symlinkJoin {
        name = "cudatoolkit";
        paths = with pkgs.cudaPackages; [
          cudatoolkit
          (pkgs.lib.getStatic cuda_cudart)
        ];
        postBuild = ''
          ln -s $out/lib $out/lib64
        '';
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.stdenv.cc ];
        shellHook = ''
          # Required for both PyTorch and Numba to find CUDA
          export CUDA_PATH=${cudatookit-with-cudart-to-lib64}
          export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

          # Required for numba-cuda Python package
          export NUMBA_CUDA_INCLUDE_PATH=${cudatookit-with-cudart-to-lib64}/include

          # cuda.bindings.nvjitlink.nvJitLinkError: ERROR_INTERNAL (6)
          # Linker error log: ERROR 4 in nvvmAddNVVMContainerToProgram, may need newer version of nvJitLink library
          export NUMBA_CUDA_USE_NVIDIA_BINDING=0

          # Required for both PyTorch and Numba, adds necessary paths for dynamic linking
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              "/run/opengl-driver" # Needed to find libGL.so, required by both PyTorch and Numba
            ]
          }:$LD_LIBRARY_PATH
        '';
      };
    };
}
