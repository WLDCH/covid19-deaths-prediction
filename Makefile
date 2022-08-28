mlflow-ui:
		mlflow ui --backend-store-uri sqlite:///mlflow.db

quality-checks:
		pipenv run black .
		pipenv run isort --profile black .
