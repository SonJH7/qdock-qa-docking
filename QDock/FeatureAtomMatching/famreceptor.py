import os
import shutil
import subprocess
import numpy as np

class Receptor():
    def __init__(self,path,pdbqt_path):
        self.name = path.split("/")[-1].split(".")[-1]
        self.pdbqt_path = pdbqt_path
        self.make_pdbqt(path,self.pdbqt_path)
        self.read_pdbqt()
        
    def make_pdbqt(self,in_path,out_path):
        if not shutil.which("prepare_receptor"):
            raise FileNotFoundError("prepare_receptor not found in PATH. Install ADFR/AutoDockTools.")
        if os.path.exists(out_path):
            self.prepare_receptor_stdout = ""
            self.prepare_receptor_stderr = ""
            return True
        cmd = ["prepare_receptor", "-r", in_path, "-o", out_path, "-A", "hydrogens"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.prepare_receptor_stdout = res.stdout
        self.prepare_receptor_stderr = res.stderr
        if res.returncode != 0:
            raise RuntimeError(
                "prepare_receptor failed with code %d: %s"
                % (res.returncode, (res.stderr or res.stdout).strip())
            )
        return os.path.exists(out_path)
    
    def read_pdbqt(self):
        autodock_atom_types = []
        f = open(self.pdbqt_path)
        for line in f.readlines():
            if line[0:6] in ["ATOM  ","HETATM"]:
                autodock_atom_types.append(line[77:79].strip())
        f.close()
        self.autodock_atom_types = np.array(autodock_atom_types) 
        
