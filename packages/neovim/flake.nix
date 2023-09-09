{

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    juuso.inputs.nixpkgs.follows = "nixpkgs";
    juuso.url = "github:jhvst/nix-config";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    nixvim.inputs.nixpkgs.follows = "nixpkgs";
    nixvim.url = "github:nix-community/nixvim";
  };

  outputs =
    inputs@{ self
    , flake-parts
    , juuso
    , nixpkgs
    , nixvim
    , ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {

      systems = nixpkgs.lib.systems.flakeExposed;
      imports = [ ];

      perSystem = { pkgs, lib, config, system, ... }: {

        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [
            inputs.juuso.overlays.default
          ];
          config = { };
        };

        packages.neovim = nixvim.legacyPackages.${system}.makeNixvimWithModule {
          inherit pkgs;
          module = {
            imports = [
              inputs.juuso.outputs.nixosModules.neovim
            ];
            extraPackages = with pkgs; [
              rustfmt
            ];
            plugins = {
              crates-nvim.enable = true;
              lsp = {
                enable = true;
                servers.rust-analyzer.enable = true;
              };
            };
            extraPlugins = [
            ];
          };
        };

        packages.default = config.packages.neovim;

      };
    };
}
