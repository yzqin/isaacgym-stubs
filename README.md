Isaac Gym Python Stubs for Code Completion
==========================================

Code completion for IsaacGym with one line `pip install`, even if without IsaacGym itself.

```bash
pip3 install git+https://github.com/yzqin/isaacgym-stubs.git
```

Depending on which IDE you are using, sometimes you may need to restart the IDE after `pip install` for re-indexing.

The magic of `stub` is that you even **do not need to pip install IsaacGym itself** to write the code.
For example, you may need to run IsaacGym on server for training but develop the code on your MacBook.
IsaacGym does not support Mac. But you can still install this repo and get smooth code completion to write IsaacGym
code on that MacBook!

### Demo

**VsCode**

![VsCode Demo](files/vscode.gif)

**PyCharm**

![PyCharm Demo](files/pycharm.gif)

### What Does This Repo Do

This repository contains the `pyi` stub for the IsaacGym library, which can be used for code completion and type
checking.
According to the guidelines outlined in [PEP-561](https://peps.python.org/pep-0561/), Python stub files contain only
type information and no runtime code.
The `stub` in this repo is generated based on IsaacGym version `1.0rc4`.
