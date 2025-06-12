{pkgs, ...}: let
  buildInputs = with pkgs; [
    cudaPackages.cuda_cudart
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    stdenv.cc.cc
    libuv
    zlib
  ];
in {
  dotenv.disableHint = true;
  devcontainer.enable = true;
  difftastic.enable = true;

  git-hooks = {
    excludes = ["^backup/"];
    hooks = {
      actionlint.enable = true;
      check-added-large-files.enable = true;
      check-json.enable = true;
      check-merge-conflicts.enable = true;
      check-yaml.enable = true;
      commitizen.enable = true;
      end-of-file-fixer.enable = true;
      eslint.enable = true;
      hadolint.enable = true;
      ripsecrets.enable = true;
      ruff.enable = true;
      ruff-format.enable = true;
      shellcheck.enable = true;
      trim-trailing-whitespace.enable = true;
      typos.enable = true;
    };
  };

  packages = with pkgs; [
    python311Packages.sounddevice
    python311Packages.pygame
    cudaPackages.cuda_nvcc
    linuxHeaders
    pre-commit
    ffmpeg
    portaudio
    libjpeg

    SDL2
    SDL2_image
    SDL2_ttf
    SDL2_mixer
    SDL2_gfx
  ];

  env = {
    C_INCLUDE_PATH = "${pkgs.linuxHeaders}/include";
    LD_LIBRARY_PATH = "${
      with pkgs;
        lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"; # For tensorflow with GPU support
    CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
  };

  languages = {
    python = {
      enable = true;
      package = pkgs.python311Full;

      venv = {
        enable = true;
      };

      uv = {
        enable = true;
        sync.enable = true;
      };
    };
  };

  enterTest = ''
    poe test
  '';

  enterShell = ''
    nvcc -V
  '';
}
