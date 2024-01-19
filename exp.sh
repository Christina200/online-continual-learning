#AGEM
nohup python general_main.py --data cifar10 --cl_type nc --agent AGEM --retrieve random --update random --mem_size 1000 --optimizer Adam --batch 4 --num_tasks 5 --fix_order True --learning_rate 0.0001 --weight_decay 0.0001 > agem_cifar10_vit_tuned.log

nohup python main_tune.py --general config/general_1.yml --data config/data/cifar10/cifar10_nc.yml --default config/agent/agem/agem_1k.yml --tune config/agent/agem/agem_tune.yml > agem_cifar10_tune.log

CUDA_VISIBLE_DEVICES=1 nohup python general_main.py --data cifar100 --cl_type nc --agent AGEM --retrieve random --update random --mem_size 1000 --optimizer Adam --batch 4 --num_tasks 10 --fix_order True --learning_rate 0.0001 --weight_decay 0.0001 > agem_cifar100_vit_tuned.log

#ER
nohup python general_main.py --data cifar10 --cl_type nc --agent ER --retrieve random --update random --mem_size 1000 --optimizer Adam --batch 4 --num_tasks 5 --fix_order True --learning_rate 0.0001 --weight_decay 0.0001> er_cifar10_vit_tuned.log

nohup python main_tune.py --general config/general_1.yml --data config/data/cifar10/cifar10_nc.yml --default config/agent/er/er_1k.yml --tune config/agent/er/er_tune.yml > er_cifar10_tune.log

#EWC
nohup python general_main.py --data cifar10 --cl_type nc --agent EWC --fisher_update_after 50 --alpha 0.9 --lambda 100 --optimizer Adam --batch 4 --num_tasks 5 --fix_order True --learning_rate 0.0001 --weight_decay 0.0001> ewc_cifar10_vit_tuned.log

CUDA_VISIBLE_DEVICES=1 nohup python main_tune.py --general config/general_1.yml --data config/data/cifar10/cifar10_nc.yml --default config/agent/ewc/ewc.yml --tune config/agent/ewc/ewc_tune.yml > ewc_cifar10_tune.log

# MIR
CUDA_VISIBLE_DEVICES=0 nohup python main_tune.py --general config/general_1.yml --data config/data/cifar10/cifar10_nc.yml --default config/agent/mir/mir_1k.yml --tune config/agent/mir/mir_tune.yml > mir_cifar10_tune.log

CUDA_VISIBLE_DEVICES=0 nohup python general_main.py --data cifar10 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 1000 --optimizer Adam --batch 4 --num_tasks 5 --fix_order True --learning_rate 0.0001 --weight_decay 0.0001 > mir_cifar10_vit_tuned.log

#GDumb
CUDA_VISIBLE_DEVICES=2 nohup python general_main.py --data cifar10 --cl_type nc --agent GDUMB --mem_size 1000 --mem_epoch 30 --minlr 0.0005 --clip 10 --optimizer Adam --batch 4 --num_tasks 5 --fix_order True --learning_rate 0.0001 --weight_decay 0.0001 > gdumb_cifar10_vit_tuned.log