{
  pkgs,
  inputs,
  config,
  lib,
  ...
}: let
  unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
in {
  options = {
    profile = lib.mkOption {
      type = lib.types.enum ["backend" "frontend" "full"];
      default = "full";
      description = "Development profile to use";
    };
  };

  config = {
    env.GREET = "devenv";

    packages = with pkgs; [
      ffmpeg
      portaudio
      unstable.supabase-cli
      redis
      zlib
    ];

    languages = {
      python = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        enable = true;
        package = pkgs.python313Full;
        directory = "./backend";

        poetry = {
          enable = true;
          activate.enable = true;
          install.enable = true;
        };
      };

      javascript = lib.mkIf (config.profile == "frontend" || config.profile == "full") {
        enable = true;
        package = pkgs.nodejs_20;
        directory = "./frontend";

        npm = {
          enable = true;
        };
        bun = {
          enable = true;
          install.enable = true;
        };
      };
    };

    git-hooks.hooks = {
      shellcheck.enable = true;
      mdsh.enable = true;
      black.enable = true;
    };

    services = lib.mkIf (config.profile == "backend" || config.profile == "full") {
      redis = {
        enable = true;
      };
    };

    processes = {
      supabase = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        exec = "supabase start";
        process-compose = {
          working_dir = "./backend";
        };
      };

      celery = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        exec = "poetry run task celery";
        process-compose = {
          working_dir = "./backend";
        };
      };

      flower = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        exec = "poetry run task flower";
        process-compose = {
          working_dir = "./backend";
        };
      };

      api = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        exec = "poetry run task api";
        process-compose = {
          working_dir = "./backend";
        };
      };

      website = lib.mkIf (config.profile == "frontend" || config.profile == "full") {
        exec = "bun --bun run dev -o";
        process-compose = {
          working_dir = "./frontend";
        };
      };
    };

    enterTest = ''
      cd backend && poetry run task test
    '';
  };
}
