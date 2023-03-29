{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs = { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustVersion = pkgs.rust-bin.stable.latest.default;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            (rustVersion.override { extensions = [ "rust-src" ]; })
          ];
          packages = with pkgs; [
            rustc
            cargo

            (pkgs.darwin.apple_sdk_11_0.callPackage <nixpkgs/pkgs/os-specific/darwin/moltenvk> {
              inherit (pkgs.darwin.apple_sdk_11_0.frameworks) AppKit Foundation Metal QuartzCore;
              inherit (pkgs.darwin.apple_sdk_11_0) MacOSX-SDK Libsystem;
              inherit (pkgs.darwin) cctools sigtool;
            })
          ];
        };
      });
}
