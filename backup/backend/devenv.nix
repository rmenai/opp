{
  pkgs,
  inputs,
  ...
}: let
  unstable = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
in {
  dotenv.disableHint = true;
  difftastic.enable = true;

  packages = with pkgs; [
    pre-commit
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

      venv = {
        enable = true;
      };

      uv = {
        enable = true;
        sync.enable = true;
      };
    };
  };

  services = {
    redis = {
      enable = true;
    };
  };

  processes = {
    supabase.exec = "poe supabase";
    celery.exec = "poe celery";
    flower.exec = "poe flower";
    api.exec = "poe api";
  };

  scripts.supabase_extract_key.exec = ''
    ./scripts/extract_supabase_key.sh
  '';

  enterShell = ''
    echo "Active Testing: $TEST"
  '';
}
