---
name: hpc-tsubame-4
description: "Usage of TSUBAME 4.0"
---

- **Do not do heavy operations on login node!** Use jobs to do so.
- Specification:
  - It consists of big nodes, 2x CPU and 4x H100. CPU-only node cpu-160 is also available. Avoid using the full node if your program only supports only single GPU.
- Credits:
  - You are allowed to use 170 (hours \* node) in total per project.
    - When using only single node, you rarely exceed this limit, so do not overthink.
    - However, be extremely careful when conducting multi-node studies, when you want to test multi-node code, start from 2 nodes, do not go up further without confidence.
- Filesystem:
  - The group directory `/gs/fs/$group` is shared, you must be **extremely careful not to edit** other people's directory. Use ``/gs/fs/$group/$user` as our own root directory, where $user is the user id.
  - Always work from group directory, not default home directory.
- SSH:
  - Url is `$user@login2.t4.gsic.titech.ac.jp`.
- The environmental variable $GSIC_USER and $GSIC_GROUP should be available, otherwise ask user to make it available.
- Job system: See https://www.t4.cii.isct.ac.jp/docs/handbook.ja/jobs/
  - **Walltime consumes credits** even if the job finishes immediately (point = (0.7 x (execution time) + 0.1 x (walltime)) x (nodes)), make sure to set the minimal walltime.
- Programs:
  - The programs inside the HPC is outdated.
  - To use Python, use via uv. If uv is not installed, install with the uv official shell script.
