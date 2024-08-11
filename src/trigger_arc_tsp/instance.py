from __future__ import annotations
import os

class Instance:
    def __init__(self) -> None:
        self.nodes = None
        self.edges = None
        self.relations = None
        self.N = None
        self.A = None
        self.R = None
        

    def load_instance_from_file(self, file_path: os.PathLike) -> None:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            self.N, self.A, self.R = map(int, lines[0].split())
            self.nodes = set(range(self.N))
            self.edges = {(x[0], x[1]) : x[2] for x in lines[1:1+self.A]}
            self.relations = {(x[0], x[1]) : x[2] for x in lines[1+self.A:1+self.A+self.R]}
        

            