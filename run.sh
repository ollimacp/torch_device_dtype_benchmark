git clone <repository_url>

cd <project_directory>

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt


python src/main.py "$@"

