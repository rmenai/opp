{
  pkgs,
  inputs,
  config,
  lib,
  ...
}: let
  unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
  pkgs-playwright = import inputs.nixpkgs-playwright {system = pkgs.stdenv.system;};
  browsers = (builtins.fromJSON (builtins.readFile "${pkgs-playwright.playwright-driver}/browsers.json")).browsers;
  chromium-rev = (builtins.head (builtins.filter (x: x.name == "chromium") browsers)).revision;
in {
  options = {
    profile = lib.mkOption {
      type = lib.types.enum ["backend" "frontend" "full"];
      default = "full";
      description = "Development profile to use";
    };
  };

  config = {
    env = {
      GREET = "devenv";

      PLAYWRIGHT_BROWSERS_PATH = "${pkgs-playwright.playwright.browsers}";
      PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS = true;
      PLAYWRIGHT_NODEJS_PATH = "${pkgs.nodejs}/bin/node";
      PLAYWRIGHT_LAUNCH_OPTIONS_EXECUTABLE_PATH = "${pkgs-playwright.playwright.browsers}/chromium-${chromium-rev}/chrome-linux/chrome";
    };

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

        venv = {
          enable = true;
        };

        uv = {
          enable = true;
          sync.enable = true;
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
        exec = "poe supabase";
        process-compose = {
          working_dir = "./backend";
        };
      };

      celery = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        exec = "poe celery";
        process-compose = {
          working_dir = "./backend";
        };
      };

      flower = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        exec = "poe flower";
        process-compose = {
          working_dir = "./backend";
        };
      };

      api = lib.mkIf (config.profile == "backend" || config.profile == "full") {
        exec = "poe api";
        process-compose = {
          working_dir = "./backend";
        };
      };

      website = lib.mkIf (config.profile == "frontend" || config.profile == "full") {
        exec = "bun --bun run dev";
        process-compose = {
          working_dir = "./frontend";
        };
      };
    };

    enterTest = ''
      cd backend && poe test
    '';
  };
}
