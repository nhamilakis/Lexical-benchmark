# Installation of extras

## polyglot

**MacOS**

- `brew install pkg-config icu4c`
- `export PATH="/opt/homebrew/opt/icu4c/bin:/opt/homebrew/opt/icu4c/sbin:$PATH"`
- `export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/opt/homebrew/opt/icu4c/lib/pkgconfig"`
- `pip install polyglot pyicu pycld2 morfessor`

**Ubuntu**

- `apt install libicu-dev`
- `pip install polyglot pyicu pycld2 morfessor`
