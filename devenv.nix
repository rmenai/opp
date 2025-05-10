{
  pkgs,
  inputs,
  ...
}: let
  unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
in {
  env.GREET = "devenv";

  packages = with pkgs; [
    ffmpeg
    portaudio
    unstable.supabase-cli
    redis
    zlib
  ];

  languages = {
    python = {
      enable = true;
      package = pkgs.python313Full;

      poetry = {
        enable = true;
        activate.enable = true;
        install.enable = true;
      };
    };
  };

  git-hooks.hooks = {
    shellcheck.enable = true;
    mdsh.enable = true;
    black.enable = true;
  };

  services = {
    redis = {
      enable = true;
    };
  };

  processes = {
    supabase = {
      exec = "supabase start";
    };

    celery = {
      exec = "poetry run task celery";
    };

    flower = {
      exec = "poetry run task flower";
    };

    api = {
      exec = "poetry run task api";
    };
  };

  enterTest = ''
    poetry run task test
  '';
}
