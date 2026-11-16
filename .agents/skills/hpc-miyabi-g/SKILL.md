---
name: hpc-miyabi-g
description: "Usage of Miyabi-G"
---

- **Do not do heavy operations on login node!** Use jobs to do so.
- Documentation:
  - The document cannot be viewed without logging in to the portal and agreeing to the rules, thus you cannot web search it.
- Specification:
  - It consists of special GH200 (120GB RAM + 96GB VRAM), up to 128 nodes may be used, connected with InfiniBand NDR200.
- Credits:
  - You are allowed to use 170 (hours \* node) in total per project.
    - When using only single node, you rarely exceed this limit, so do not overthink.
    - However, be extremely careful when conducting multi-node studies, when you want to test multi-node code, start from 2 nodes, do not go up further without confidence.
- Filesystem:
  - Only use user directory inside work directory `/work/$group/$user`, where $group is the group name without `g` postfix, $user is the user id of syntax `mXXXXX` where XXXXX is 5-digit integer.
- Connection:
  - Use ssh to connect to `$user@miyabi-g.jcahpc.jp`. Use something like tmux for persistent session.
  - After initial SSH login, OTP is requested. OTP is refreshed every 30 seconds. Since you are too slow to enter it, the user should be using ControlMaster and already connecting using SSH, so that you don't need to enter it. If not, ask the user to do so.
- Programs:
  - The programs inside the HPC is outdated.
  - To use Python, use via uv. If uv is not installed, install with the uv official shell script.
- Job System:
  - Submit `$script_path=*.sh` file using `qsub -W group_list=$group $script_path`
  - Options can be also in the file as comment with PBS prefix:
    ```sh
    #!/bin/sh
    #PBS -q debug-g
    ```
  - Options:
    - Options other than `q` and `W` are optional.
    - `-q`: QUEUE name (required)
      - There are mig nodes, which is 1/4 node.
      - Check the specs using `qstat --rsc -x`.
      - On first login, run `qstat --rscuse` to check which QUEUE name is less crowded.
    - `-W group_list=`: group name (required)
    - `-o (filename)`: log file
    - `-e (filename)`: error file
    - `-j oe`: output log and error to standard output
    - `-N`: job name
    - `-m abe`: send mail
    - `-V`: inherit environmental variables
    - `-r y`/`-r n`: job can/cannot be re-run
      `-l select=(num)`: number of nodes
      `-l walltime=(walltime)`: walltime, nothing to do with credits (unlike Tsubame) but may make job execution faster\*\*
    - `-l mpiprocs=`: required when mpi, the total number of mpi process will be number of nodes x mpiprocs
    - `-l ompthreads=`: not required
    - `-l pcore=`: core file size per process
  - Append `--help` tp check usage.
  - `qstat`, `qdel` may be used to check and cancel jobs.
- The user may be storing hpc-specific information at `hpc` directory directly under project directory.
