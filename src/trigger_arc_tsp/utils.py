import os

current_dir = os.path.dirname(os.path.abspath(__file__))

INSTANCES_DIR = os.path.join(current_dir, "..", "..", "instances")
SOLUTIONS_DIR = os.path.join(current_dir, "..", "..", "solutions")
MODELS_DIR = os.path.join(current_dir, "..", "..", "models")


def cleanup_instance_name(instance: str) -> str:
    if not instance.endswith(".txt"):
        error_msg = "Instance file must be a .txt file"
        raise ValueError(error_msg)
    if instance.startswith("instances/"):
        instance = instance[10:]

    if instance.count("/") != 1:
        instance = f"instances_release_1/{instance}"

    # check if file exists
    if not os.path.exists(os.path.join(INSTANCES_DIR, instance)):
        error_msg = f"Instance file {instance} not found"
        raise FileNotFoundError(error_msg)

    return instance
