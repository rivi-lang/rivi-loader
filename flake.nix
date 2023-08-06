{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/23.05";
    naersk.url = "github:nix-community/naersk";
  };
  outputs = { self, nixpkgs, flake-utils, naersk, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        naersk' = pkgs.callPackage naersk { };
        pkgs = import nixpkgs { inherit system; };
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
        devShells.default = pkgs.mkShell {
          RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
          inherit nativeBuildInputs;
          packages = with pkgs; [
            rustc
            cargo

            spirv-tools
            spirv-cross
          ];
        };
        packages = rec {
          default = capabilities;
          capabilities = naersk'.buildPackage {
            inherit nativeBuildInputs;

            pname = "capabilities";
            src = ./.;

            LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
            overrideMain = old: {
              preConfigure = ''
                cargo_build_options="$cargo_build_options --example capabilities"
              '';
            };
          };
        };
      });
}
