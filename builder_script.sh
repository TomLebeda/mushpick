#NOTE: if you run just `cargo build --release` it will build for the platform you are currently on
cargo build --release --target x86_64-unknown-linux-gnu;  # build for Linux
cargo build --release --target x86_64-pc-windows-gnu; # build for Windows

# copy the binaries into <PROJECT_ROOT>/compiled/<binary>
cp ./target/x86_64-unknown-linux-gnu/release/mushpick ./compiled/mushpick
cp ./target/x86_64-pc-windows-gnu/release/mushpick.exe ./compiled/mushpick.exe
