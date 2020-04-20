# lynx

Code for forecasting constraints on primordial gravitational waves. 

## Description

`lynx` provides tools to apply map-space component separation to observations of the polarized CMB. 

## Installation

In order to set up the necessary environment:

1. create an environment `lynx` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate lynx
   ```
3. install `lynx` with:
   ```
   python setup.py install # or `develop`
   ```

Then take a look into the `scripts` and `notebooks` folders.

## Note

This project has been set up using PyScaffold 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
