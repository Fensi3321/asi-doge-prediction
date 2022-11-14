# asi project

1. Make sure you are using Python 3.9

```bash
python3 --version
```

2. Set up your virtual environment

```bash
python3 -m venv venv
```

3. Activate venv

    3.1 Unix

    Depending on your Shell
    ```bash
    source venv/bin/activate(.fish/.csh/.ps1)
    ```

    3.2 Windows

    PoweShell
    ```bash
    c:\ venv\Scripts\Activate.ps1
    ```

    cmd.exe
    ```bash
    c:\ venv\Scripts\activate.bat
    ```

4. Install Kedro

    ```bash
    pip install kedro
    ```

5. Install requirements

```bash
pip install -r src/requirements.txt
``` 

```bash
kedro build-reqs
```

6. Download Data

    5.1 If you are using POSIX shell you can use `download_data.sh` script

    else go to [Yahoo Finance](https://finance.yahoo.com/quote/DOGE-USD/history?p=DOGE-USD) and manually download the data. Rename the file as `doge-usd.csv` and place it under `./data/01_raw/` path.


7. Test run

    Command should run successfully
```bash
kedro run
```
