# Body Brain Behavior - Quality Diversity

## Install guide
Clone the repo and its submodules and `cd` into it
```shell
git clone --recurse-submodules https://github.com/giorgia-nadizar/BBB-QD.git
cd BBB-QD
```
If you have already cloned the repository without its submodules, run the following command to fix it
```shell
git submodule update --init --recursive
```

### Normal usage
Install some required libraries
```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
sudo apt-get install cmake
sudo apt-get install freeglut3-dev
sudo apt-get install libglfw3
sudo apt-get install libgl1-mesa-glx
```
Create conda environment and install a required library
```shell
conda env create -f environment.yml
conda activate bbbqd
conda install -c conda-forge libstdcxx-ng -y
```
Build and install package
```shell
python setup.py install
```

### Docker
The Docker file is `bbbqd.Dockerfile`
To execute the project with docker build the file first
```shell
docker build --target run-image -t bbb-qd:0.0.1 -f bbbqd.Dockerfile .
```
and the run it 
```shell
docker run --gpus all -it -v "$(pwd)/results:/app/results" bbb-qd:0.0.1
```
the flag `--gpus` is needed to allow the usage of the GPU, if available; the flag `-v` enables the mounting on the
container of the folder `results`.
They can both be removed if needed.

To run the container in interactive mode add the flag `--entrypoint bash`.


### Testing installation
To test your installation was successful you can run `examples/install_tester.py` which will test that both `jax` and
`evogym` are properly installed. This script will print a bunch of arrays.

Otherwise, you can run `examples/viz_tester.py`, which will launch an evogym simulation and its visualization.


## Experiments, results, videos
The folder `experiments` contains the code for running all the experiments performed so far.
Namely, `cgp_ga.py` evolves a CGP controller with both global and modular controllers for 4 fixed bodies.
Instead, `cgp_body_ga.py` evolves the body-within a 5x5 grid-and the CGP controller for both global and modular
controllers.

The analysis of results is in the folder `analysis`, in the `preliminary.ipynb` notebook.

Some videos of evolved robots are available in the `videos` folder.