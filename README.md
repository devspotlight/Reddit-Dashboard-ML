# Bot Identification

There is brief information about how this should be run and how to get involved in the project.

Do a local clone:

```bash
git clone https://github.com/TopengineerOrg/botidentification.git
cd botidentification
git checkout comments/dataset

```

## Create a dir inside the `lib` folder nameth `data`

```bash
mkdir lib
cd lib
mkdir data
```

Now download comments dataset from (https://drive.google.com/file/d/122jWdX3ma2UBpz3eJzNj_2JUcfE4BNhs/view) in your new `data` folder.

## Installation

First steps before begin with the installations. Make sure to get installed the next libraries:

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

## Getting started

Clean comment dataset

```bash
python clean_data.py
```

Run the Model

```bash
python useridentification.py
```


## More...

Detailed documentation about this project can be found inside `docs/`.



