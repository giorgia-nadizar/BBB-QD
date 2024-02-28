# Body Brain Behavior - Quality Diversity

## Install guide

### Normal usage
Install some required libraries
```
sudo apt-get install xorg-dev libglu1-mesa-dev
```
Create conda environment and install a required library
```
conda env create -f environment.yml
conda activate bbbqd
conda install -c conda-forge libstdcxx-ng -y
```
Build and install package
```
python setup.py install
```


### Docker
Docker file will be available.
