{
  inputs = {
    devenv.url = "github:cachix/devenv";
    flake-parts.url = "github:hercules-ci/flake-parts";
    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";
    naersk.url = "github:nix-community/naersk";
    neovim.url = "github:rivi-lang/rivi-loader?dir=packages/neovim";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";
    nix2container.url = "github:nlewo/nix2container";
    nixpkgs.url = "github:nixos/nixpkgs/23.05";
  };
  outputs =
    inputs@{ self
    , devenv
    , flake-parts
    , flake-utils
    , naersk
    , nixpkgs
    , ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {

      systems = nixpkgs.lib.systems.flakeExposed;
      imports = [
        inputs.devenv.flakeModule
      ];

      perSystem = { pkgs, lib, config, inputs', ... }:

        let
          naersk' = pkgs.callPackage naersk { };
          nativeBuildInputs = with pkgs; [
            pkgconfig
            vulkan-loader
          ] ++ lib.optionals stdenv.hostPlatform.isDarwin [
            libiconv
            (pkgs.darwin.apple_sdk_11_0.callPackage "${toString self.inputs.nixpkgs}/pkgs/os-specific/darwin/moltenvk" {
              inherit (pkgs.darwin.apple_sdk_11_0.frameworks) AppKit Foundation Metal QuartzCore;
              inherit (pkgs.darwin.apple_sdk_11_0) MacOSX-SDK Libsystem;
              inherit (pkgs.darwin) cctools sigtool;
            })
          ];

        in
        {

          packages.neovim = inputs'.neovim.packages.default;
          packages.capabilities = naersk'.buildPackage rec {
            inherit nativeBuildInputs;

            pname = "capabilities";
            src = ./.;

            LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";

            overrideMain = old: {
              preConfigure = ''
                cargo_build_options="$cargo_build_options --example ${pname}"
              '';
            };
          };
          packages.gpus = naersk'.buildPackage rec {
            inherit nativeBuildInputs;

            pname = "gpus";
            src = ./.;

            LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";

            overrideMain = old: {
              preConfigure = ''
                cargo_build_options="$cargo_build_options --example ${pname}"
              '';
            };
          };

          packages.default = config.packages.capabilities;

          devenv.shells.default = {

            env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";

            packages = with pkgs; [
              rustc
              cargo

              spirv-tools
              spirv-cross

            ] ++ config.packages.default.nativeBuildInputs;
          };

        };
    };
}
