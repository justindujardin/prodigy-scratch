
If you don’t have [homebrew](https://brew.sh), install it:

```sh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Then install python3 and virtualenv:

```sh
brew install python3
sudo easy_install pip
sudo pip install --upgrade virtualenv 
```

Then we're going to [create a virtualenv with Python3](https://www.tensorflow.org/install/install_mac#installing_with_virtualenv) inside of the root project folder, and activate the environment:

```sh
# Setup the virtual env with python v3
virtualenv --system-site-packages -p python3 ./
# activate the isolated python env to make `python` available on the commandline
source ./bin/activate 
# Install tensorflow, etc.
pip install -r requirements.txt
```
