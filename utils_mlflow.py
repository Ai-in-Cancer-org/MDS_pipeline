import mlflow, time, traceback
from mlflow.entities.run_info import RunInfo
from mlflow.entities import RunStatus

class MlflowTracker:
    active_run: mlflow.ActiveRun = None
    tracking_uri = "mlruns/"
    experiment_name = ""
    run_uuid = ""
    run_name = ""

    @staticmethod
    def initialize(experiment_name, run_name="", run_uuid=None):
        MlflowTracker.experiment_name = experiment_name
        MlflowTracker.run_uuid = run_uuid
        MlflowTracker.run_name = run_name

    @staticmethod
    def connect():
        mlflow.set_tracking_uri(MlflowTracker.tracking_uri)
        mlflow.set_experiment(MlflowTracker.experiment_name)
        if MlflowTracker.run_uuid is not None:
            MlflowTracker.active_run = mlflow.start_run(run_id=MlflowTracker.run_uuid)
        else:
            MlflowTracker.active_run = mlflow.start_run(run_name=MlflowTracker.run_name)
        run_info: RunInfo = MlflowTracker.active_run.info
        MlflowTracker.run_uuid = run_info.run_id

    @staticmethod
    def reconnect():
        mlflow.end_run()
        mlflow.set_tracking_uri(MlflowTracker.tracking_uri)
        mlflow.set_experiment(MlflowTracker.experiment_name)
        MlflowTracker.active_run = mlflow.start_run(run_id=MlflowTracker.run_uuid)

    @staticmethod
    def finish():
        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

    @staticmethod
    def fail():
        mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

def retry(cb, max_retries=5, err_msg=None):
    i, exception = 0, None
    while(i<max_retries):
        i+=1
        try:
            return cb()
        except Exception as e:
            exception = e
            print(traceback.format_exc())
            time.sleep(10)
    raise exception
