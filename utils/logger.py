import csv
import json
import os
import time


class RunLogger:
    def __init__(self, run_dir, config=None, resume=False):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        if config is not None:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)
        self._fields = None
        self._fh = None
        self._writer = None
        self._resume = resume and os.path.exists(self.csv_path)
        self.t0 = time.time()

    def log(self, row):
        row = {"wall_s": round(time.time() - self.t0, 2), **row}
        if self._writer is None:
            self._fields = list(row)
            new = not self._resume
            self._fh = open(self.csv_path, "a" if self._resume else "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            if new:
                self._writer.writeheader()
        self._writer.writerow({k: row.get(k, "") for k in self._fields})
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()


def read_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        out[k] = vals
    return out
