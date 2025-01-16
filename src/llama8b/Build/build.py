import mlflow
import os
import logging
import subprocess

def build_inference(run_id):
    os.environ["container"] = "docker"
    path = '/git/clarus-llama8b-dag/build_docker'
    endpoint = 'registry-docker-registry.registry.svc.cluster.local:5001/llama8b:prod'


    def download_artifacts(run_id, path):
        mlflow.set_tracking_uri("http://mlflow-tracking.mlflow.svc.cluster.local:5000")

        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=path)

        # Buscar el archivo model.pkl y moverlo a la carpeta local_path en caso de que se encuentre en una subcarpeta
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.startswith("model"):
                    logging.info(f"Encontrado archivo model.pkl en: {root}")
                    os.rename(os.path.join(root, file), os.path.join(local_path + '/model', file))
                elif file.startswith("requirements"):
                    logging.info(f"Encontrado archivo requirements.txt en: {root}")
                    os.rename(os.path.join(root, file), os.path.join(path, file))

    def modify_requirements_file(path):
        required_packages = ["fastapi", "uvicorn", "pydantic", "numpy"]

        with open(f"{path}/requirements.txt", "r") as f:
            lines = f.readlines()

        with open(f"{path}/requirements.txt", "w") as f:
            for line in lines:
                if line.strip() not in required_packages:
                    f.write(line)

            f.write("\n")

            for package in required_packages:
                f.write(f"{package}\n")


    logging.warning(f"Downloading artifacts from run_id: {run_id['best_run']}")
    download_artifacts(run_id['best_run'], path)
    modify_requirements_file(path)

    args = [
        "/kaniko/executor",
        f"--dockerfile={path}/Dockerfile",
        f"--context={path}",
        f"--destination={endpoint}",
        f"--cache=false",
        f"--single-snapshot"
    ]
    result = subprocess.run(
        args,
        check=True  # Lanza una excepción si el comando devuelve un código diferente de cero
    )
    logging.warning(f"Kaniko executor finished with return code: {result.returncode}")