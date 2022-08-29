mlflow-ui:
		mlflow ui --backend-store-uri sqlite:///mlflow.db
		
setup:
		pip install pipenv
		pipenv install
		
streamlit-dashboard:
		streamlit run dashboard/app.py

quality-checks:
		pipenv run black .
		pipenv run isort --profile black .
