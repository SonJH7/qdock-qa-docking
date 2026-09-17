# -*- coding: utf-8 -*-
"""
@author: Jinyin Zha
"""
#Basic Packages
import os
import sys
import copy
import numpy as np
import prody
import json
#QC simulator
import neal 
from pyqubo import Binary, Constraint
#Receprot and Ligand
sys.path.append(os.path.abspath(os.path.realpath(__file__))[:-8])
from famreceptor import Receptor
from famligand import Ligand
from fam_cqm import build_fam_cqm

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
             save_qubo=True,sim_dock=True,save_match=True,save_pose=True):
        if self.__step != "dock":
            print("Please run make_%s first!"%(self.__step))
            return
        return [self.indiv_dock(ligand,edge_cutoff,K_dist,K_mono,n_pos,
                                save_qubo,sim_dock,save_match,save_pose) \
                for ligand in self.ligands]
            
    #Step 4, Docking(single)
    def indiv_dock(self,ligand,edge_cutoff,K_dist,K_mono,n_pos=30,
                   save_qubo=True,sim_dock=True,save_match=True,save_pose=True): # edge_cutoff serves as c_dist
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

                # E_dist is defined only for i < k.
                # Same-ligand-atom conflicts are handled by C_atom_exact.
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
        sampler = neal.SimulatedAnnealingSampler()
        raw_solution = sampler.sample_qubo(qubo,num_reads=n_pos,seed=42)
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
                    "sampler": type(sampler).__name__,
                    "time_limit": sampler_kwargs.get("time_limit"),
                    "dist_constraints": dist_constraints,
                    "constraint_limit_hit": limit_hit,
                }
                with open("CQM/%s_sampler_meta.json"%(ligand.name), "w") as f:
                    json.dump(meta, f, indent=2)

        # Convert matches back to docking pose
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
            trans = prody.superpose(ls,gs)[1]
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
        os.makedirs("pocs", exist_ok=True) # generate feature atoms with AutoSite
        # remove stale pseudo feature file from previous fallback runs
        try:
            if os.path.exists("pocs/pseudo_fp_energy.pdb"):
                os.remove("pocs/pseudo_fp_energy.pdb")
        except Exception:
            pass
        os.system("autosite -r %s --boxcenter [%1.3f,%1.3f,%1.3f]\
                  --boxdim [%d,%d,%d] -o pocs"%(
            self.receptor.pdbqt_path,
            self.box_center[0],self.box_center[1],self.box_center[2],
            self.dims[0]-1,self.dims[1]-1,self.dims[2]-1))
        pocs = []
        poc_files = os.listdir("pocs") # build feature_atoms from PDB files containing _fp_
        for poc_file in poc_files:
            if "_fp_" not in poc_file or poc_file.startswith("pseudo_"):
                continue
            pocs.append(prody.parsePDB("pocs/%s"%(poc_file)))

        # Fallback: generate energy-based pseudo-features if AutoSite produces none
        if not pocs:
            fallback_enabled = os.environ.get("QDOCK_FAM_FALLBACK", "1") == "1"
            if not fallback_enabled:
                raise RuntimeError("AutoSite produced no feature points.")
            pocs = [self.__fallback_features_from_autogrid()]

        self.feature_atoms = pocs[0]
        if len(pocs) > 1:
            for i in range(1,len(pocs)):
                self.feature_atoms += pocs[i]
        self.box_Dmatrix = prody.buildDistMatrix(self.feature_atoms)
        self.__step = "dock"

    def __fallback_features_from_autogrid(self):
        import shutil
        import subprocess
        import numpy as np
        import math

        if not shutil.which("autogrid4"):
            raise RuntimeError("autogrid4 not found in PATH. Install ADFR/AutoDockTools.")

        fallback_n = int(os.environ.get("QDOCK_FAM_FALLBACK_N", "30"))
        fallback_n = max(1, fallback_n)
        use_quota = os.environ.get("QDOCK_FAM_FALLBACK_QUOTA", "0") == "1"

        def compute_quota(counts, total_n):
            types = [t for t, c in counts.items() if c > 0]
            if not types or total_n <= 0:
                return {}
            if total_n < len(types):
                types_sorted = sorted(types, key=lambda t: counts[t], reverse=True)
                return {t: 1 for t in types_sorted[:total_n]}
            total = float(sum(counts.values()))
            raw = {t: (total_n * counts[t] / total) for t in types}
            quota = {t: max(1, int(math.floor(raw[t]))) for t in types}
            remaining = total_n - sum(quota.values())
            if remaining > 0:
                frac = sorted(types, key=lambda t: raw[t] - math.floor(raw[t]), reverse=True)
                for t in frac:
                    if remaining <= 0:
                        break
                    quota[t] += 1
                    remaining -= 1
            elif remaining < 0:
                frac = sorted(types, key=lambda t: raw[t] - math.floor(raw[t]))
                for t in frac:
                    if remaining >= 0:
                        break
                    if quota[t] > 1:
                        quota[t] -= 1
                        remaining += 1
            return quota

        quota = None
        if use_quota:
            counts = {}
            for lig in getattr(self, "ligands", []):
                for t in getattr(lig, "autodock_atom_types", []):
                    counts[t] = counts.get(t, 0) + 1
            quota = compute_quota(counts, fallback_n)

        # run autogrid4 to build energy maps
        info = """outlev 1 
npts %d %d %d
gridfld %s.fld
spacing %1.5f
receptor_types %s
ligand_types %s
receptor %s
gridcenter %1.3f %1.3f %1.3f
smooth 0.500000
"""%(self.dims[0]-1,self.dims[1]-1,self.dims[2]-1,
self.receptor.name,self.grid_length,
" ".join(self.receptor_autodock_atom_types.tolist()),
" ".join(self.ligands_autodock_atom_types.tolist()),self.receptor.pdbqt_path,
self.box_center[0],self.box_center[1],self.box_center[2])
        for t in self.ligands_autodock_atom_types:
            info += "map Receptor.%s.map\n" % (t,)

        gpf_path = "%s_fam_fallback.gpf" % self.receptor.name
        with open(gpf_path, "w") as f:
            f.write(info)
        res = subprocess.run(["autogrid4", "-p", gpf_path, "-l", "%s_fam_fallback.log" % self.receptor.name],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError("autogrid4 failed: %s" % (res.stderr or res.stdout).strip())

        # read grid start coord
        xyz_path = "%s.xyz" % self.receptor.name
        if not os.path.exists(xyz_path):
            raise RuntimeError("autogrid4 did not produce %s" % xyz_path)
        with open(xyz_path) as f:
            box_st = np.array([float(i.split()[0]) for i in f.readlines()])

        # collect candidate points across map types
        candidates = []
        for t in self.ligands_autodock_atom_types:
            map_path = "Receptor.%s.map" % t
            if not os.path.exists(map_path):
                continue
            with open(map_path) as f:
                es = np.array([float(i) for i in f.readlines()[6:]])
            if es.size == 0:
                continue
            if use_quota:
                q = quota.get(t, 0) if quota else 0
                if q <= 0:
                    continue
                idxs = np.argsort(es)[:q]
            else:
                idxs = np.argsort(es)[:fallback_n]
            for idx in idxs:
                candidates.append((es[idx], t, int(idx)))

        if not candidates:
            raise RuntimeError("No energy map points available for fallback.")

        candidates.sort(key=lambda x: x[0])
        if not use_quota:
            candidates = candidates[:fallback_n]

        # map index to coordinate and build pseudo feature PDB
        def idx_to_coord(idx):
            k = idx % self.dims[0]
            j = (idx // self.dims[0]) % self.dims[1]
            i = idx // (self.dims[0] * self.dims[1])
            return (
                box_st[0] + k * self.grid_length,
                box_st[1] + j * self.grid_length,
                box_st[2] + i * self.grid_length,
            )

        def map_type_to_elem(t):
            elem = t
            if elem not in self.w_dict:
                elem = elem[0]
            if elem == "A":
                elem = "C"
            return elem

        pseudo_path = os.path.join("pocs", "pseudo_fp_energy.pdb")
        with open(pseudo_path, "w") as f:
            for idx, (energy, t, grid_idx) in enumerate(candidates, start=1):
                x, y, z = idx_to_coord(grid_idx)
                elem = map_type_to_elem(t)
                atom_name = elem
                line = "ATOM  %5d %-4s FPF A%4d    %8.3f%8.3f%8.3f  1.00  0.00" % (
                    idx, atom_name, idx, x, y, z
                )
                line = line.ljust(76) + ("%2s" % elem)
                f.write(line + "\n")
            f.write("END\n")

        return prody.parsePDB(pseudo_path)


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
