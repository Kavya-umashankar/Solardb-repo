## Requirements and Setting up
# 2.1 Setting up using Windows Command prompt
Each of these project require different python versions to run, some of them are python 3.6.x, python 3.8.x, python 3.9.x.
Install the required python version from https://www.python.org/downloads/

Next step is to create an virtual environment to install all the required package. 
In command prompt, navigate to the desidered directory and the command for creating virtual environment: 
>python -m venv name_of_virtualenv 
If the name of virtual environment is 'envr', then navigate to envr\Scripts in command prompt and run 'activate' to activate the virtual environment.

install all the required modules from requirements.txt by running the following command: 
>pip install -r requirements.txt.

Next, to install jupyter notebook in this environment by : 
>pip install jupyter notebook.

To add this virtual envoirnment kernel to the jupyter notebook : 
>ipython kernel install --name "envr" --user. 

Launch jupyter notebook by typing in jupyter notebook in command prompt.
>jupyter notebook

In jupyter notebook, under kernel --> Change kernel --> envr(virtual enviornment)
and run the cells


# 2.2 Setting up using Miniconda
There are different versions of Miniconda for different python versions available on
https://docs.conda.io/en/latest/miniconda.html

use either windows command prompt terminal or the miniconda terminal to set up an
environment. To create the environment: where env1 is the environment name.
>conda create --name env1 python=3.X

To activate the environment: 
>conda activate env1

If there is need to set up gpu, then:
>conda install -c anaconda tensorflow-gpu

Navigate to the folder location and install the requirements from the text file. 
It can be done in two ways: using conda and using pip.
Using conda command: 
>conda install --file requirements.txt

Using pip:
Install pip in miniconda and then install requirements through pip:
Install pip: 
>conda install pip
Install requirements: 
>pip install -r requirements.txt


Next, Iistall the ipykernel: 
>conda install -c anaconda ipykernel

Add kernel to jupyter: 
>python -m ipykernel -user -name=env1

In jupyter notebook, under kernel --> Change kernel --> env1(virtual enviornment)
and run the cells
