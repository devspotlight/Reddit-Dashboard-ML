# Machine Learning API to Identify Bots and Trolls

This code can be used to identify bots and trolls on Reddit. It's part of a two part blog series the first covering how to build a moderator dashboard and the second covering this machine learning model. The dashboard repository can be found here https://github.com/devspotlight/Reddit-Dashboard

There is a brief guide on how to run the code. There are notebooks used for training a model and code to run a Flask server with the predictions API. It also includes Procfile so you can quickly run the API on Heroku.

First, do a local clone:

```bash
git clone https://github.com/devspotlight/botidentification.git
cd botidentification
git checkout comments/dataset

```

## Download the training data

Create a dir inside the `lib` folder named `data` 

```bash
mkdir lib
cd lib
mkdir data
```

Now download the comments dataset as a CSV file into your new `data` folder. You can download the bots and trolls training data [here](https://drive.google.com/file/d/1FDvHMLbJ8mXlsiiNnLgFCV6Yom1m_xbU/view?usp=sharing). For normal user training data, you can dump the data from Redshift to a CSV file.

## Installation

Please first install Python 3 and Jupyter.

### Python 3

To see which version of Python 3 you have installed, open a command prompt and run

```bash
python3 --version
```
If you are using Ubuntu 16.10 or newer, then you can easily install Python 3.6 with the following commands:

```bash
sudo apt-get update
sudo apt-get install python3
```
If you’re using another version of Ubuntu (e.g. the latest LTS release), we recommend using the deadsnakes PPA to install Python 3.6:

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
```

### Virtual environment

We can create a virtualenv on both Linux and OS X by running python3 -m venv myvenv. It will look like this:

```bash
python3 -m venv myvenv
```
Start your virtual environment by running:

```bash
source myvenv/bin/activate
```

### Requirements or Python packages

```bash
pip install -r requirements.txt
```

### Jupyter

Follow the (Jupyter installation instructions](https://jupyter.org/install). You can then open it by running:

```
jupyter notebook
```


## Running the code

Clean comment dataset 
* `clean_data_bots_trolls.ipynb`
* `clean_data_normies.ipynb`

Build the model:
* `user_identification.ipynb`

Run the Flask API

```bash
export FLASK_ENV=development
export FLASK_APP=app.py
flask run
```



