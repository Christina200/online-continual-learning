#AGEM
nohup python general_main.py --data cifar10 --cl_type nc --agent AGEM --retrieve random --update random --mem_size 5000 --optimizer Adam --batch 4 --num_tasks 5 --fix_order True > agem_cifar10_vit.log

nohup python main_tune.py --general config/general_1.yml --data config/data/cifar10/cifar10_nc.yml --default config/agent/agem/agem_1k.yml --tune config/agent/agem/agem_tune.yml > agem_cifar10_tune.log