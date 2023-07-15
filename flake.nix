{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
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
          (pkgs.darwin.apple_sdk_11_0.callPackage "${toString self.inputs.nixpkgs}/pkgs/os-specific/darwin/moltenvk" {
            inherit (pkgs.darwin.apple_sdk_11_0.frameworks) AppKit Foundation Metal QuartzCore;
            inherit (pkgs.darwin.apple_sdk_11_0) MacOSX-SDK Libsystem;
            inherit (pkgs.darwin) cctools sigtool;
          })
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs;
          packages = with pkgs; [
            rustc
            cargo

            spirv-tools
            spirv-cross
          ];
        };
        defaultPackage = naersk'.buildPackage {
          inherit nativeBuildInputs;

          pname = "capabilities";
          version = "0.1.0";
          src = ./.;

          LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
          overrideMain = old: {
            preConfigure = ''
              cargo_build_options="$cargo_build_options --example capabilities"
            '';
          };
        };
      });
}
