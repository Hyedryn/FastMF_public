import os
import sys
#%%
import fastmf.generation as gen
from fastmf.reports import evaluator

base_path = r"C:\Users\quent\Documents\Github\FastMF_python\tests\dataPaper"
run_id = 0
orientation_estimate = "CSD"
seed = 111
print("Seed: ", seed)
task_name = "paperStLucGE"

# Synthetizer Path
synthetizer_file = os.path.join(base_path, "synthetizer", "type-structured", "raw",
                                f"type-structured_task-{task_name}_run-{run_id}_raw.pickle")

# MF Generation
genStructured = gen.Generator(synthetizer_file, base_path, orientation_estimate_sh_max_order=12,
                              orientation_estimate=orientation_estimate, recompute_S0mean=False, compute_vf=False,
                              compute_swap=False)
genStructured.computeExhaustiveMF(processes=1)

