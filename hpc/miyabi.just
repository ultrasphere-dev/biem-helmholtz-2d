# User name
user:="${MIYABI_USER}"
# Group name
group:="${MIYABI_GROUP}"
# The remote server to connect to
remote:=user + "@miyabi-g.jcahpc.jp"
# Your directory in the group drive
group_dir:="/work" / group / user
# The path to the script to be executed
script_path:="miyabi.sh"

default: ssh

# connect to group drive (from local)
ssh:
    ssh -t {{remote}} "cd {{group_dir}}; bash --login"

# run the script (from hpc)
run:
    git pull
    qsub -W group_list={{group}} {{script_path}}
