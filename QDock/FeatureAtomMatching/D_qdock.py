# -*- coding: utf-8 -*-
"""
@author: Jinyin Zha
"""
#Basic Packages
import os
import sys
import copy
import json
from datetime import datetime
import numpy as np
import prody
#QC simulator
import neal 
from pyqubo import Binary, Constraint
#Receprot and Ligand
sys.path.append(os.path.abspath(os.path.realpath(__file__))[:-8])
from famreceptor import Receptor
from famligand import Ligand
from fam_cqm import build_fam_cqm

def _save_sampler_meta(path, meta):
    try:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)
    except Exception:
        pass

def report_docking_analysis(docked_poses, native_path, original_ligand):
    if not os.path.exists(native_path):
        print("[Error] Native PDB file not found: %s" % (native_path))
        return None

    import pandas as pd

    native = prody.parsePDB(native_path)
    native_coords = native.getCoords()
    orig_coords = original_ligand.coords

    results_data = []
    for i, pose_coords in enumerate(docked_poses):
        rmsd = prody.calcRMSD(native_coords, pose_coords)
        res = prody.superpose(orig_coords, pose_coords)
        mapped_orig = res[0]
        shape_diff = prody.calcRMSD(mapped_orig, pose_coords)
        is_feasible = shape_diff < 0.1
        is_success = rmsd <= 2.0
        results_data.append({
            "Pose_ID": i + 1,
            "RMSD(A)": round(rmsd, 4),
            "Shape_Diff(A)": round(shape_diff, 8),
            "Feasible": is_feasible,
            "Success(<=2.0A)": is_success
        })

    df = pd.DataFrame(results_data)
    total = len(df)
    feasible_count = df["Feasible"].sum() if total else 0
    success_count = df["Success(<=2.0A)"].sum() if total else 0

    print("\n Detailed analysis results (%d samples)" % (total))
    print("-" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    if total:
        print("1. Feasibility Rate (structural validity) : %d/%d (%.1f%%)" % (
            feasible_count, total, (feasible_count / total) * 100))
        print("2. Sampling Success Rate (<=2.0A) : %d/%d (%.1f%%)" % (
            success_count, total, (success_count / total) * 100))
        print("3. Minimum RMSD (mRMSD)           : %.4f A" % (df["RMSD(A)"].min()))
        print("4. Average RMSD                   : %.4f A" % (df["RMSD(A)"].mean()))
    print("=" * 60)
    return df

class FAMDock():
    def __init__(self):
        self.__step = "receptor"
        self.w_dict = {'H': 2.2, # electronegativity
         'C': 2.55,
         'N': 3.04,
         'O': 3.44,
         'F': 3.98,
         'Si': 1.9,
         'P': 2.19,
         'S': 2.58,
         'Cl': 3.16,
         'As': 2.18,
         'Se': 2.48,
         'Br': 2.96,
         'I': 2.66,
         'B': 2.04} 
        
    def get_step(self):
        return self.__step
    
    #Step 1, Prepare receptor file. 
    def make_receptor(self,receptor_path):
        if receptor_path[-4:] != ".pdb":
            print("Receptor shoud be PDB format!")
            return
        self.receptor = Receptor(receptor_path,"receptor.pdbqt")
        self.receptor_autodock_atom_types = np.unique(
            self.receptor.autodock_atom_types)
        self.__step = "ligand"
        
    #Step 2, Prepare Ligand files to be docked.     
    def make_ligand(self,ligand_paths):
        if self.__step != "ligand":
            print("Please run make_%s first!"%(self.__step))
            return
        self.ligands = []
        if not os.path.exists("Ligands"):
            os.mkdir("Ligands")
        for path in ligand_paths:
            self.ligands.append(Ligand(path,"Ligands"))
        self.ligands_autodock_atom_types = np.unique(np.hstack(
            [i.autodock_atom_types for i in self.ligands]))
        self.__step = "box"
    
    #Step 3, Create Docking Box with a ligand
    def make_box_ligand(self,path,
                           center_length=8,grid_length=1.0):
        if self.__step != "box":
            print("Please run make_%s first!"%(self.__step))
            return
        if not os.path.exists("Box_Rawligand"):
            os.mkdir("Box_Rawligand")
        self.raw_ligand = Ligand(path,"Box_Rawligand")
        self.box_center = np.mean(self.raw_ligand.coords,axis=0)
        self.box_lengths = center_length + 2 * np.max(np.abs(
            self.raw_ligand.coords - self.box_center),axis=0)
        self.grid_length = grid_length
        self.__autosite()
        
    #Step 3, Create Docking Box with manual input of center coordiante and lengths
    def make_box_input(self,x,y,z,dx,dy,dz,grid_length=1.0):
        self.box_center = np.array([x,y,z])
        self.box_lengths = np.array([dx,dy,dz])
        self.grid_length = grid_length
        self.__autosite()
    
    #Step 4, Docking (serial)
    def dock(self,edge_cutoff,K_dist,K_mono,n_pos=30,
             save_qubo=True,sim_dock=True,save_match=True,save_pose=True,
             sampler=None,sampler_kwargs=None,save_sampler_meta=True):
        if self.__step != "dock":
            print("Please run make_%s first!"%(self.__step))
            return
        return [self.indiv_dock(ligand,edge_cutoff,K_dist,K_mono,n_pos,
                                save_qubo,sim_dock,save_match,save_pose,
                                sampler,sampler_kwargs,save_sampler_meta) \
                for ligand in self.ligands]
            
    #Step 4, Docking(single)
    def indiv_dock(self,ligand,edge_cutoff,K_dist,K_mono,n_pos=30,
                   save_qubo=True,sim_dock=True,save_match=True,save_pose=True,
                   sampler=None,sampler_kwargs=None,save_sampler_meta=True): # edge_cutoff serves as c_dist
        if self.__step != "dock":
            print("Please run make_%s first!"%(self.__step))
            return
        #4.1 Create QUBO
        H = 0
        Vs = [] # build the x_ij candidate list
        #4.1.1 Make Vertexes (quadratic term in QUBO)
        for i in range(ligand.n):
            t_lig = ligand.autodock_atom_types[i]
            if t_lig not in self.w_dict.keys():
                t_lig = t_lig[0]
            if t_lig == "A":
                t_lig = "C"
            for j,fs in enumerate(self.feature_atoms): # N feature points g_s; iterate over every feature atom
                t_fs = fs.getElement()
                weight = abs(self.w_dict[t_lig] - self.w_dict[t_fs]) - 0.5 # map the compatibility weight w_{a_i g_s}
                x = Binary("%d_%d_%5.5f"%(i,j,weight)) # binary variable x_ij matching atom i to point j
                Vs.append([x,i,j,weight])
                H += weight * x**2 # implement the weight term directly in the energy
        if len(Vs)*(len(Vs)-1)/2 > 100000000:
            print("Too Big!!!")
            return
        #4.1.2, Build paper-consistent constraints
        C_atom_exact = 0
        for i in range(ligand.n):
            row_vars = [x for x, ii, _, _ in Vs if ii == i]
            C_atom_exact += (1 - sum(row_vars)) ** 2

        C_feature_unique = 0
        C_dist = 0

        for p in range(len(Vs)):
            x1, i1, j1, _ = Vs[p]

            for q in range(p + 1, len(Vs)):
                x2, i2, j2, _ = Vs[q]

                # Same-ligand-atom conflicts are handled by exact-one.
                if i1 == i2:
                    continue

                # Different ligand atoms cannot use the same feature.
                if j1 == j2:
                    C_feature_unique += x1 * x2

                dd = abs(
                    ligand.d_matrix[i1, i2]
                    - self.box_Dmatrix[j1, j2]
                )
                if dd > edge_cutoff:
                    C_dist += x1 * x2

        C_mono = C_atom_exact + C_feature_unique

        H += K_mono * Constraint(C_mono, label='mono')
        H += K_dist * Constraint(C_dist, label='link')
        model = H.compile() # compile to QUBO       
        qubo, offset = model.to_qubo()
        #4.1.3, Save QUBO Models
        if not os.path.exists("QUBOs"):
            os.mkdir("QUBOs")
        if save_qubo:
            np.save("QUBOs/%s.npy"%(ligand.name),qubo)
        #4.2 Docking by PyQUBO simulator (simulated annealing)
        if not sim_dock:
            return 
        sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)
        sampler_kwargs_original = dict(sampler_kwargs)
        num_reads = sampler_kwargs.pop("num_reads", n_pos)
        if sampler is None:
            sampler = neal.SimulatedAnnealingSampler()
            sampler_kwargs.setdefault("seed", 42)
            if num_reads is None:
                num_reads = n_pos
        if num_reads is None:
            raw_solution = sampler.sample_qubo(qubo,**sampler_kwargs)
        else:
            raw_solution = sampler.sample_qubo(qubo,num_reads=num_reads,**sampler_kwargs)
        if save_sampler_meta:
            if not os.path.exists("QUBOs"):
                os.mkdir("QUBOs")
            meta = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sampler": type(sampler).__name__,
                "num_reads": num_reads,
                "sampler_kwargs": sampler_kwargs_original,
            }
            if hasattr(sampler, "get_meta"):
                meta.update(sampler.get_meta())
            elif hasattr(raw_solution, "info"):
                meta["info"] = raw_solution.info
            _save_sampler_meta(os.path.join("QUBOs","%s_sampler_meta.json"%(ligand.name)), meta)
        samples = []
        for sample in raw_solution.samples():
            #if sample not in samples:
                samples.append(sample)
        #4.3, Convert matches back to docking pose
        news = [] # store transformed coordinates, one pose per sample
        match = []
        for sample in samples:
            ls = []
            gs = []
            ws = []
            this_match = []
            for info in filter(lambda x:1== x[1],sample.items()): # extract only active matches with x_ij = 1
                i,j,w = info[0].split("_")
                i = int(i)
                j = int(j)
                w = float(w)
                ls.append(ligand.coords[i]) # ligand coordinate r_i^st 
                gs.append(self.feature_atoms.getCoords()[j]) # target coordinate R_si^st
                ws.append(w)
                this_match.append(info[0])
            try: # stack ls and gs; skip samples without enough matches for Kabsch alignment
               ls = np.vstack(ls)
               gs = np.vstack(gs)
               match.append(copy.deepcopy(this_match))
            except:
               continue
            trans=prody.superpose(ls,gs)[1] # Kabsch alignment from ls to gs; the transform contains k_rot and b
            news.append(prody.applyTransformation(trans,ligand.coords)) # apply the transform to all ligand coordinates
        if not os.path.exists("Matches"):
            os.mkdir("Matches")
        if save_match and match:
            np.save("Matches/%s_match.npy"%(ligand.name),match) 
        if not os.path.exists("Poses"):
                os.mkdir("Poses")
        if save_pose and news:
            tmp = ligand.ligand.copy() # create a PDB pose file; add coordinate sets for multiple poses
            tmp.setCoords(news[0])
            if len(news) > 1:
                tmp.addCoordset(np.array(news[1:]))
            prody.writePDB("Poses/%s_poses.pdb"%(ligand.name),tmp)
        return np.array(news)

    #Step 4, Docking with CQM + D-Wave Hybrid (serial)
    def dock_cqm(self,edge_cutoff,K_dist=None,K_mono=None,n_pos=30,
                 time_limit=5,max_constraints=None,
                 save_samples=True,save_match=True,save_pose=True,
                 sampler=None,sampler_kwargs=None,save_sampler_meta=True):
        if self.__step != "dock":
            print("Please run make_%s first!"%(self.__step))
            return
        return [self.indiv_dock_cqm(ligand,edge_cutoff,K_dist,K_mono,n_pos,
                                    time_limit,max_constraints,
                                    save_samples,save_match,save_pose,
                                    sampler,sampler_kwargs,save_sampler_meta) \
                for ligand in self.ligands]

    #Step 4, Docking(single) with CQM + D-Wave Hybrid
    def indiv_dock_cqm(self,ligand,edge_cutoff,K_dist=None,K_mono=None,n_pos=30,
                       time_limit=5,max_constraints=None,
                       save_samples=True,save_match=True,save_pose=True,
                       sampler=None,sampler_kwargs=None,save_sampler_meta=True):
        if self.__step != "dock":
            print("Please run make_%s first!"%(self.__step))
            return

        if max_constraints is None:
            env_limit = os.environ.get("QDOCK_CQM_MAX_CONSTRAINTS")
            if env_limit:
                try:
                    max_constraints = int(env_limit)
                except Exception:
                    max_constraints = None
        if time_limit is None:
            env_time = os.environ.get("QDOCK_CQM_TIME_LIMIT")
            if env_time:
                try:
                    time_limit = float(env_time)
                except Exception:
                    time_limit = None

        cqm, vertices, dist_constraints, limit_hit = build_fam_cqm(
            ligand,
            self.feature_atoms,
            self.box_Dmatrix,
            self.w_dict,
            edge_cutoff,
            K_dist=K_dist,
            K_mono=K_mono,
            max_constraints=max_constraints,
        )

        if len(vertices) * (len(vertices) - 1) / 2 > 100000000:
            print("Too Big!!!")
            return

        cfg_backend_params = {}
        cfg_sampler_kwargs = {}
        cfg_path = os.environ.get("QDOCK_CQM_CONFIG")
        if cfg_path and os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                cfg_backend_params = cfg.get("backend_params", {}) or {}
                cfg_sampler_kwargs = cfg.get("sampler_kwargs", {}) or {}
            except Exception:
                cfg_backend_params = {}
                cfg_sampler_kwargs = {}

        if sampler is None:
            try:
                from dwave.system import LeapHybridCQMSampler
            except Exception as exc:
                print("LeapHybridCQMSampler not available: %s" % exc)
                return
            sampler = LeapHybridCQMSampler(**cfg_backend_params)

        sampler_kwargs = {} if sampler_kwargs is None else dict(sampler_kwargs)
        if cfg_sampler_kwargs:
            merged = dict(cfg_sampler_kwargs)
            merged.update(sampler_kwargs)
            sampler_kwargs = merged
        if time_limit is not None and "time_limit" not in sampler_kwargs:
            sampler_kwargs["time_limit"] = time_limit

        if not hasattr(sampler, "sample_cqm"):
            print("Sampler does not support sample_cqm")
            return

        sampleset = sampler.sample_cqm(cqm, **sampler_kwargs)
        if hasattr(sampleset, "record") and "is_feasible" in sampleset.record.dtype.names:
            feasible = sampleset.record["is_feasible"]
            if np.any(feasible):
                sampleset = sampleset.filter(lambda d: d.is_feasible)

        samples = []
        energies = []
        for sample, energy in zip(sampleset.samples(), sampleset.record.energy):
            samples.append({k: int(v) for k, v in sample.items()})
            energies.append(float(energy))
        if n_pos is not None:
            samples = samples[:n_pos]
            energies = energies[:n_pos]

        if save_samples:
            if not os.path.exists("CQM"):
                os.mkdir("CQM")
            payload = {
                "samples": samples,
                "energies": energies,
                "dist_constraints": dist_constraints,
                "constraint_limit_hit": limit_hit,
            }
            with open("CQM/%s_samples.json"%(ligand.name), "w") as f:
                json.dump(payload, f, indent=2)
            if save_sampler_meta:
                meta = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "sampler": type(sampler).__name__,
                    "time_limit": sampler_kwargs.get("time_limit"),
                    "sampler_kwargs": dict(sampler_kwargs),
                    "dist_constraints": dist_constraints,
                    "constraint_limit_hit": limit_hit,
                }
                if hasattr(sampler, "get_meta"):
                    meta.update(sampler.get_meta())
                _save_sampler_meta(os.path.join("CQM","%s_sampler_meta.json"%(ligand.name)), meta)

        news = []
        match = []
        for sample in samples:
            ls = []
            gs = []
            this_match = []
            for info in filter(lambda x:1== x[1],sample.items()):
                parts = info[0].split("_", 1)
                if len(parts) < 2:
                    continue
                i = int(parts[0])
                j = int(parts[1])
                ls.append(ligand.coords[i])
                gs.append(self.feature_atoms.getCoords()[j])
                this_match.append(info[0])
            try:
                ls = np.vstack(ls)
                gs = np.vstack(gs)
                match.append(copy.deepcopy(this_match))
            except:
                continue
            trans=prody.superpose(ls,gs)[1]
            news.append(prody.applyTransformation(trans,ligand.coords))
        if not os.path.exists("Matches"):
            os.mkdir("Matches")
        if save_match and match:
            np.save("Matches/%s_match.npy"%(ligand.name),match) 
        if not os.path.exists("Poses"):
            os.mkdir("Poses")
        if save_pose and news:
            tmp = ligand.ligand.copy()
            tmp.setCoords(news[0])
            if len(news) > 1:
                tmp.addCoordset(np.array(news[1:]))
            prody.writePDB("Poses/%s_poses.pdb"%(ligand.name),tmp)
        return np.array(news)
    
    
    def __autosite(self):
        self.dims = (self.box_lengths / self.grid_length).astype(np.int)
        for i in range(3):
            if self.dims[i] % 2 == 0:
                self.dims[i] = self.dims[i] - 1
        os.system("mkdir pocs") # generate feature atoms with AutoSite
        os.system("autosite -r %s --boxcenter [%1.3f,%1.3f,%1.3f]\
                  --boxdim [%d,%d,%d] -o pocs"%(
            self.receptor.pdbqt_path,
            self.box_center[0],self.box_center[1],self.box_center[2],
            self.dims[0]-1,self.dims[1]-1,self.dims[2]-1))        
        pocs = []
        poc_files = os.listdir("pocs") # build feature_atoms from PDB files containing _fp_
        for poc_file in poc_files:
            if "_fp_" not in poc_file:
                continue
            else:
                pocs.append(prody.parsePDB("pocs/%s"%(poc_file)))
        self.feature_atoms = pocs[0]
        if len(pocs) > 1:
            for i in range(1,len(pocs)):
                self.feature_atoms += pocs[i] 
        self.box_Dmatrix = prody.buildDistMatrix(self.feature_atoms)
        self.__step = "dock"


'''
D (docking box) and discretization interval: grid_length sets the resolution;
box_center and box_lengths determine the box center and size.
(line 67)

$g_{s_i}$ (discretized spatial point):
FAM uses AutoSite-generated feature_atoms as $g_{s_i}$.
(line 203, line 213)

$D_{s_i s_j}$ (distance between points): FAM uses
box_Dmatrix = buildDistMatrix(feature_atoms).
(line 213)
'''
