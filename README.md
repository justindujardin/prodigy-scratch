
If you don’t have [homebrew](https://brew.sh), install it:

```sh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Then install python3 and virtualenv:

```bash
brew install python3
sudo easy_install pip
sudo pip install --upgrade virtualenv 
```

[create a virtualenv with Python3](https://www.tensorflow.org/install/install_mac#installing_with_virtualenv) inside of the root project folder, and activate the environment:

```bash
virtualenv --system-site-packages -p python3 ./
source ./bin/activate 
pip install -r requirements.txt
```

Then install your prodigy whl from file

```bash
pip install ./prodigy-[etc-your-file].whl
```

Finally, run one of the examples from the project root:
```bash
sh recipes/textcat_attention_weights.sh
```