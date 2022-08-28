mlflow-ui:
		mlflow ui --backend-store-uri sqlite:///mlflow.db

quality-cheks:
		pipenv run black .
		pipenv run isort .
