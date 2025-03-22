## ▶️ Getting started with Poetry:
1. Install ```pipx```:

* Mac installation:
```bash
brew install pipx
```

* Windows installation:
```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

2. Using ```pipx``` install ```poetry``` (with export plugin)
```bash
pipx install poetry
```

3. Restart the terminal and run:
```bash
poetry self add poetry-plugin-export
```

4. Navigate to the project's root folder:
```bash
cd deep-generative-models-biu
```

5. Run poetry install:
```bash
poetry install
```

* Before moving on to next sections, please make sure that `poetry install` is finished without installation errors.

6. Clone `lost-in-the-middle` original repo:
```bash
git clone https://github.com/nelson-liu/lost-in-the-middle.git
```

7. Run `ExperimentRunner` on `test` to make sure everything was set up correctly:
```bash
poetry run python main.py --experiment test
```

## ⚙️ Set poerty as virtual environment (venv):

1. Make sure that poetry has an active venv:
```bash
poetry env info
```

* Result should look like the following:
```bash
Virtualenv
Python:         3.12.8
Implementation: CPython
Path:           /Users/user/Library/Caches/pypoetry/virtualenvs/inference-6Rb_cK8F-py3.12
Executable:     /Users/user/Library/Caches/pypoetry/virtualenvs/inference-6Rb_cK8F-py3.12/bin/python
Valid:          True

Base
Platform:   darwin
OS:         posix
Python:     3.12.8
Path:       /opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12
Executable: /opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12
```

* If your result looks like the following please continue to read the `*` steps, if your result is valid you can skip to `2`.
```bash
Virtualenv
Python:         3.12.8
Implementation: CPython
Path:           NA
Executable:     NA
```

* Run the following command using the python executable path from the valid `Base` section example:
```bash
poetry env use <Base Executable>
```

* Run again the info command and make sure that the result is now valid.

2. Activate poetry's venv:
```bash
source $(poetry env info --path)/bin/activate
```

3. Get venv python dir to set as python interpreter (OPTIONAL - relevant only if you want to run `py` files directly using IDE's interpreter):
```bash
poetry env info --path
```

* Setting the python intepreter is different in different IDEs, in all of them you need to open the environment setting manager and enter the previous section path.