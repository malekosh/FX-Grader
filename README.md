# FX-Grader

## Dependncies and recommended installation instructions

### Windows 10
1) download and install anaconda: https://docs.anaconda.com/anaconda/install/windows/
2) from anaconda prompt, download the python packages listed in the requirements.txt file
3) install ipywidgets using conda prompt:
```
conda install -c conda-forge ipywidgets

```
4) Launch jupyterlab from Anaconda Navigator.

if you have installed the latest anacoda version this should be enough

### MAC OS
1) download and install anaconda: https://docs.anaconda.com/anaconda/install/mac-os/
2) Install nodejs and Ipywidgets using:
```
conda install nodejs -c conda-forge --repodata-fn=repodata.json

conda install -c conda-forge ipywidgets

```
3) check jupyterlab version using the command `jupyter --version`
If jupyterlab is less than version 3.x, type the following command:
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
4) install dependencies from the requirements.txt file using `conda install`.
5) Launch Jupyter-lab from Anaconda Navigator.


### Ipywidgets installation documentation
If you still have trouble getting Ippwidgets running, please refer to the installation documentation: 

https://ipywidgets.readthedocs.io/en/latest/user_install.html


