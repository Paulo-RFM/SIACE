#!/usr/bin/env python3
"""Treina classificadores binarios "uma especie contra as demais", no mesmo protocolo da bananeira.

Para cada alvo (padrao: palmeira = 2, mangueira = 4):
  python train_species_efficientnet.py --binary-target <id> --species-names species_names.json --run-id <run>
  python compare_binary_vs_multiclass.py --binary-run runs/<run>
O treino e identico ao run classif_20261103_115814_efficientnet_binaria_bananeira (mesmo script, mesma
semente 42, mesmos recortes anotados, mesma divisao espacial, selecao por F1-macro de validacao,
teste avaliado uma unica vez pelo proprio script). Nada e sobrescrito: aborta se o run ja existir.
Log de cada treino: runs/<run>/train_stdout.log. Rodar a partir de lab_2026/.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LAB = Path(__file__).resolve().parent
NAMES = {2: "palmeira", 3: "arvore_seca", 4: "mangueira", 5: "bananeira", 6: "jabuticabeira"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs = []
    for t in args.targets:
        run_id = f"classif_{stamp}_efficientnet_binaria_{NAMES[t]}"
        run_dir = LAB / "runs" / run_id
        if run_dir.exists():
            raise SystemExit(f"{run_dir} ja existe; nao sobrescrevo.")
        cmd = [sys.executable, "train_species_efficientnet.py", "--binary-target", str(t),
               "--species-names", "species_names.json", "--run-id", run_id]
        print(f"[{datetime.now():%H:%M:%S}] treino {NAMES[t]}: {' '.join(cmd)}", flush=True)
        tmp_log = LAB / "runs" / f".{run_id}.log"
        with open(tmp_log, "w") as log:
            subprocess.run(cmd, cwd=LAB, stdout=log, stderr=subprocess.STDOUT, check=True)
        tmp_log.rename(run_dir / "train_stdout.log")
        subprocess.run([sys.executable, "compare_binary_vs_multiclass.py", "--binary-run", f"runs/{run_id}"],
                       cwd=LAB, check=True)
        runs.append(run_id)
        print(f"[{datetime.now():%H:%M:%S}] concluido {run_id}", flush=True)
    (LAB / "runs" / f"binary_suite_{stamp}.json").write_text(json.dumps({"runs": runs, "targets": args.targets},
                                                                         indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
