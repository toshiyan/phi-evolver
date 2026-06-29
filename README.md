# A tool to compute evolution of phi

`phi-evolver` is a Python package for computing evolution of phi by solving the following equation:

$$
\phi'' + 2{\cal H}\phi' + a^2V_{,\phi} = 0
$$

# Installation

The installation builds the pure-Python module:

```bash
python -m pip install "phi-evolver @ git+https://github.com/toshiyan/phi-evolver.git@main"
```

* `class_aux/`
  containing necessary files to run CLASS

* `class_output/`
  if you do not specify any output directory, the output background.dat and logfile are saved in this directory

* `example/`
  storing example files for running this software

