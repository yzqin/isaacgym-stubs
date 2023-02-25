Isaac Gym Python Stubs for Code Completion
==========================================

Enable code completion for IsaacGym simply with `pip install isaacgym-stubs`, even without IsaacGym itself!

```bash
# Install from PyPi
pip3 install isaacgym-stubs

# Alternatively, install from Github
# pip3 install git+https://github.com/yzqin/isaacgym-stubs.git
```

The magic of `stub` is that you even **do not need to pip install IsaacGym itself**.
For example, you may need to run IsaacGym on server for training but develop the code on the MacBook.
IsaacGym may not support Mac. But you can still install this repo and get smooth code completion to write IsaacGym
code on that MacBook!

Depending on which IDE you are using, sometimes you may need to restart the IDE after `pip install` for re-indexing.

### Demo

**VsCode**

![VsCode Demo](files/vscode.gif)

**PyCharm**

![PyCharm Demo](files/pycharm.gif)

### Overview

This repository contains the `pyi` stub for the IsaacGym library, which can be used for code completion and type
checking.
According to the guidelines outlined in [PEP-561](https://peps.python.org/pep-0561/), Python stub files contain only
type information and no runtime code.
The `stub` in this repo is generated based on IsaacGym version `1.0rc4`.
