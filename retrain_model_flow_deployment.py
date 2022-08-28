from train_model import main

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=main,
    name="train_model_flow",
    schedule=CronSchedule(cron='0 20 * * *', timezone='Europe/Paris'),
    work_queue_name="train_model_queue",
)

deployment.apply()
