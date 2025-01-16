from datetime import datetime
from airflow.decorators import dag, task
from kubernetes.client import models as k8s
from airflow.models import Variable

@dag(
    description='Esto es una demo de CT1',
    schedule_interval=None, 
    start_date=datetime(2025, 1, 15),
    catchup=False,
    tags=['demo', 'CT1'],
)
def Demo_CT1():

    env_vars={
        "POSTGRES_USERNAME": Variable.get("POSTGRES_USERNAME"),
        "POSTGRES_PASSWORD": Variable.get("POSTGRES_PASSWORD"),
        "POSTGRES_DATABASE": Variable.get("POSTGRES_DATABASE"),
        "POSTGRES_HOST": Variable.get("POSTGRES_HOST"),
        "POSTGRES_PORT": Variable.get("POSTGRES_PORT"),
        "TRUE_CONNECTOR_EDGE_IP": Variable.get("CONNECTOR_EDGE_IP"),
        "TRUE_CONNECTOR_EDGE_PORT": Variable.get("IDS_EXTERNAL_ECC_IDS_PORT"),
        "TRUE_CONNECTOR_CLOUD_IP": Variable.get("CONNECTOR_CLOUD_IP"),
        "TRUE_CONNECTOR_CLOUD_PORT": Variable.get("IDS_PROXY_PORT"),
        "MLFLOW_ENDPOINT": Variable.get("MLFLOW_ENDPOINT"),
        "MLFLOW_TRACKING_URI": Variable.get("MLFLOW_ENDPOINT"),
        "MLFLOW_TRACKING_USERNAME": Variable.get("MLFLOW_TRACKING_USERNAME"),
        "MLFLOW_TRACKING_PASSWORD": Variable.get("MLFLOW_TRACKING_PASSWORD")
    }

    volume_mount = k8s.V1VolumeMount(
        name="dag-dependencies", mount_path="/git"
    )

    init_container_volume_mounts = [
        k8s.V1VolumeMount(mount_path="/git", name="dag-dependencies")
    ]

    volume = k8s.V1Volume(name="dag-dependencies", empty_dir=k8s.V1EmptyDirVolumeSource())

    init_container = k8s.V1Container(
        name="git-clone",
        image="alpine/git:latest",
        command=["sh", "-c", "mkdir -p /git && cd /git && git clone -b main --single-branch https://github.com/MiKeLFernandeZz/TFG_Demo.git"],
        volume_mounts=init_container_volume_mounts
    )
    
    pod_spec = k8s.V1Pod(
        api_version='v1',
        kind='Pod',
        spec=k8s.V1PodSpec(
            runtime_class_name='nvidia',  # Establecer runtimeClassName a 'nvidia'
            containers=[
                k8s.V1Container(
                    name='base',
                )
            ]
        )
    )

    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='read',
        task_id='read',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={'cpu': '0.5'},
            limits={'cpu': '0.5'}
        ),
        priority_class_name='medium-priority',
        env_vars=env_vars
    )
    def read_task():
        import sys
        import redis
        import uuid
        import pickle
    
        sys.path.insert(1, '/git/TFG_Demo/src/redwine')
        from Data.read_data import read_data
        
        """
        MODIFY WHAT YOU WANT
        """
        
        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,
            password='pass'
        )
    
        df = read_data()
        
        df_id = str(uuid.uuid4())
        redis_client.set(df_id, pickle.dumps(df))
    
        return df_id
    
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='process',
        task_id='process',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={'cpu': '0.5'},
            limits={'cpu': '0.5'}
        ),
        priority_class_name='medium-priority',
        env_vars=env_vars
    )
    def process_task(df_id):
        import sys
        import redis
        import uuid
        import pickle
    
        sys.path.insert(1, '/git/TFG_Demo/src/redwine')
        from Process.data_processing import data_processing
        
        """
        MODIFY WHAT YOU WANT
        """
        
        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,
            password='pass'
        )
        df = pickle.loads(redis_client.get(df_id))
    
        dp = data_processing(df)
        
        dp_id = str(uuid.uuid4())
        redis_client.set(dp_id, pickle.dumps(dp))
    
        return dp_id
    
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='svc',
        task_id='svc',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={'cpu': '1.5'},
            limits={'cpu': '1.5'}
        ),
        priority_class_name='medium-priority',
        env_vars=env_vars
    )
    def svc_task(dp_id):
        import sys
        import redis
        import uuid
        import pickle
    
        sys.path.insert(1, '/git/TFG_Demo/src/redwine')
        from Models.SVC_model_training import svc_model_training
        
        """
        MODIFY WHAT YOU WANT
        """
        
        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,
            password='pass'
        )
        dp = pickle.loads(redis_client.get(dp_id))
    
        return svc_model_training(dp)
        
        
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='elastic',
        task_id='elastic',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={'cpu': '1.5'},
            limits={'cpu': '1.5'}
        ),
        priority_class_name='medium-priority',
        env_vars=env_vars
    )
    def elastic_task(dp_id):
        import sys
        import redis
        import uuid
        import pickle
    
        sys.path.insert(1, '/git/TFG_Demo/src/redwine')
        from Models.ElasticNet_model_training import elasticNet_model_training
        
        """
        MODIFY WHAT YOU WANT
        """
        
        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,
            password='pass'
        )
        dp = pickle.loads(redis_client.get(dp_id))
    
        return elasticNet_model_training(dp)
        
        
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='select_best',
        task_id='select_best',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={'cpu': '0.5'},
            limits={'cpu': '0.5'}
        ),
        priority_class_name='medium-priority',
        env_vars=env_vars
    )
    def select_best_task():
        import sys
        sys.path.insert(1, '/git/TFG_Demo/src/redwine')
        from Deployment.select_best_model import select_best_model
        
        """
        MODIFY WHAT YOU WANT
        """
        
        run_id = select_best_model()
        
        return run_id
    
    @task.kubernetes(
        image='clarusproject/dag-image:kaniko',
        name='build',
        task_id='build',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={'cpu': '1'},
            limits={'cpu': '2.5'}
        ),
        priority_class_name='medium-priority',
        env_vars=env_vars
    )
    def build_task(run_id):
        import sys
        sys.path.insert(1, '/git/TFG_Demo/src/redwine')
        from Build.build import build_inference
        
        """
        MODIFY WHAT YOU WANT
        """
        
        return build_inference(run_id)
        
        
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='redis_clean',
        task_id='redis_clean',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        env_vars=env_vars
    )
    def redis_clean_task(ids):
        import redis
    
        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,
            password='pass'
        )
    
        redis_client.delete(*ids)
    
    read_result = read_task()
    process_result = process_task(read_result)
    svc_result = svc_task(process_result)
    elastic_result = elastic_task(process_result)
    select_best_result = select_best_task()
    build_result = build_task(select_best_result)
    redis_task_result = redis_clean_task([read_result, process_result])
    
    # Define the order of the pipeline
    read_result >> process_result >> [svc_result, elastic_result] >>  select_best_result >> build_result >> redis_task_result
# Call the DAG 
Demo_CT1()