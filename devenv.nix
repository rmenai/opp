{pkgs, ...}: {
  env.GREET = "devenv";

  packages = [pkgs.git pkgs.ffmpeg];

  languages = {
    python = {
      enable = true;

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
}
