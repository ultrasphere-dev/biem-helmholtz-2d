#!/bin/sh
#PBS -q debug-g
#PBS -l walltime=00:03:00
#PBS -o out.log
#PBS -j oe
#PBS -m abe
#PBS -l select=1
#PBS -l mail_power_info=true

cd ${PBS_O_WORKDIR}
cd biem-helmholtz-2d
git pull
uv sync --all-extras
uv run pytest
