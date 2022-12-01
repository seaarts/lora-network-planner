# loraplan - LoRa Network Planing tools

.. image:: https://github.com/shunsvineyard/python-sample-code/workflows/Test/badge.svg
    :target: https://github.com/shunsvineyard/python-sample-code/actions?query=workflow%3ATest

<p align="left">
<a href="https://github.com/seaarts/lora-network-planner/actions?query=workflow%3ATest"><img alt="Tests" src=https://github.com/seaarts/lora-network-planner/workflows/Test/badge.svg>
<a href="https://github.com/seaarts/lora-network-planner/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

LoRa network planning utilities and tools in python.

## Notes on making a package.
- To continue with the packaging tutorial go to [this step](https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives)

- Read about [versioning](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#pre-release-versioning)

- See [PyPA](https://www.pypa.io/en/latest/)

- Also see [PythonPackages](https://py-pkgs.org/06-documentation)

## Documentation
Use `sphinx` to generate documentation. I prefer `NumPy` style - see [styleguide](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). For a gentle tutorial on `sphinx` see [shunsvineyard](https://shunsvineyard.info/2019/09/19/use-sphinx-for-python-documentation/).

Use `sphinx-apidoc -f -o source ../src/loraplan/` in `docs` to make `.rst` docfiles. Then use `make html` to refresh your docs.


See the `sphinx-proofs` extension that you may want for fun. [Link](https://github.com/executablebooks/sphinx-proof)


## See notes on using Colab and github
Check out this [Colab demo](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=3VQqVi-3ScBC).

It seems possible to maintain a folder of noteoboks, and to create a link that opens a notebook in colab.

## Testing
- Read about [pytest](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#test-discovery)

## Additional learning.

- Best practices on strucure and testing on [Ionel MC's blog](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure%3E)
 
