### How to generate the `pyi` file:

```bash
pip3 install mypy
stubgen -m gym_38 # If you are using Python 3.8

```

Then rename the `gym_38.pyi` as `gymapi.pyi` and place it in the stub directory. You may also need some manual label to
clean up the auto-generated stub.
