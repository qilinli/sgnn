# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=0 python gns/multi_scale/train_multi_scale.py --mode=train --data_path=./datasets/taylor_impact_2d/data_processed/ --model_path=./models/Taylor_impact_2d/ --output_path=./rollouts/Taylor_impact_2d/ --batch_size=1 --noise_std=0.001 --connection_radius=0.1 --layers=5 --hidden_dim=64 --lr_init=0.001 --ntraining_steps=10000 --lr_decay_steps=3000 --dim=2 --project_name=Segment-3D --run_name=NS1e-3_R0.015_L5N64 --nsave_steps=1000 --log=False

# Rollout
CUDA_VISIBLE_DEVICES=0 python -m gns.train --mode=rollout --data_path=/home/jovyan/share/EPIMETHEUS-LOCAL/8TB-share/share/qilin/gns_data/Concrete2D-T-Step2/ --model_path=./models/Taylor_impact_2d/NS1e-3_R0.015_L5N64/ --model_file=model-010000.pt --output_path=./rollouts/Taylor_impact_2d/ --batch_size=1 --noise_std=0.001 --connection_radius=0.015 --layers=5 --hidden_dim=64 --dim=2 --project_name=Segment-3D --run_name=nsNS1_R15_L5N64  --log=False

# Visualisation
python -m gns.render_rollout_2d_T --rollout_path=rollouts/Taylor_impact_2d/rollout_0.pkl --output_path=rollouts/Taylor_impact_2d/rollout_0.gif


# Notes
- If net config changed before evaluation, load weights may fail
- If subtle config changed, evaluation may have low results
- Train loss (acc) and val loss (pos) are not comparable currently
- wandb step increase by default everytime wandb.log() is called
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise)
- pytorch geometric caps the knn in radius_graph to be <=32
- The original domain is x (-165, 165) and y (-10, 85). Normalise it to (0,1) and (0,1) will change the w/h ratio. 
- Be careful with the simulation domain, the Bullet in impact loading has made y too large unnessarily
