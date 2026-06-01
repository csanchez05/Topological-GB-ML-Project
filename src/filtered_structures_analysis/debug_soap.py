import os
import sys
from viable_soap_extractor import collect_jobs, process_structure

jobs = collect_jobs()
print(f"Total jobs: {len(jobs)}")

for i, job in enumerate(jobs):
    try:
        process_structure(job)
    except Exception as e:
        print(f"Failed at job {i}: {e}")

print("Done.")
