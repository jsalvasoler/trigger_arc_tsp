import os
from math import floor
from random import random

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


def fisher_yates_shuffle(list_: list) -> list:
    amnt_to_shuffle = len(list_)
    # We stop at 1 because anything * 0 is 0 and 0 is the first index in the list
    # so the final loop through would just cause the shuffle to place the first
    # element in... the first position, again.  This causes this shuffling
    # algorithm to run O(n-1) instead of O(n).
    while amnt_to_shuffle > 1:
        # Indice must be an integer not a float and floor returns a float
        i = int(floor(random() * amnt_to_shuffle))  # no cryptographic purpose
        # We are using the back of the list to store the already-shuffled-indice,
        # so we will subtract by one to make sure we don't overwrite/move
        # an already shuffled element.
        amnt_to_shuffle -= 1
        # Move item from i to the front-of-the-back-of-the-list. (Catching on?)
        list_[i], list_[amnt_to_shuffle] = list_[amnt_to_shuffle], list_[i]
    return list_
