Isaac Gym Python Stubs for Code Completion
==========================================

### Repository Overview

This repository contains the pyi stub for the IsaacGym library, which can be used for code completion and type checking.
According to the guidelines outlined in [PEP-561](https://peps.python.org/pep-0561/), Python stub files contain only
type information and no runtime code.

### Installation

To install the repository, use pip and then your IDE or editor will automatically provide code completion.
The `stub` in this repo is generated based on IsaacGym version `1.0rc4`. Depending on which IDE you are using, sometimes
you may need to restart the IDE after `pip install` to re-indexing the IsaacGym library.

```bash
pip3 install git+https://github.com/yzqin/isaacgym-stubs.git
```

### Demo

**VsCode**
![VsCode Demo](files/vscode.gif)


**PyCharm**
![PyCharm Demo](files/pycharm.gif)


### Troubleshooting

If you are a PyCharm/CLion user and still cannot achieve code completion after performing the pip install, you could
consider installing the original IsaacGym library by using `pip install .` instead of `pip install -e .` in the
`IsaacGym_Preview_4_Package/isaacgym/python/` directory.
