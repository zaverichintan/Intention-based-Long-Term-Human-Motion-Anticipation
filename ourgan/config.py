import torch
from os.path import join, isdir, isfile, abspath, dirname
import pathlib

class config(object):

    """Training Configurations"""
    device = torch.device("cuda")

    # number of input and output frames
    n_in = 10
    n_out = 20
    n_frames = n_in + n_out
    # params for the model
    hidden_units = 1024
    # hidden_units = 512 # amass

    noise_dim = 32
    label_dim = 8

    framrates = [25]

    dropout = 0

    train_batchsize = 64
    test_batchsize = 1024
    
    model_run = 10
    model_seed = 0
    force_new_training = False

    MAX_EPOCH = 100

    teacher_forcing = 1
    warmup_frames = 3

    lambda_term = 10
    forecast = 10
    weight_clipping_term = 1

    loss_decay = False
    # loss_decay = True
    decay_factor = 0.5
    epochs_for_euclidean = 20

    end_frame_decoder = 15

    data_dim = None

    stacks = 3

    local_path = abspath(pathlib.Path(__file__).parent.absolute())
    activity_dir = abspath(join(local_path, '../data/clustered_labels'))+'/'

    # name of the dataset
    dataset_name = 'H36M' #{H36M, CMU, H36M_less_joints, H36M_less_joints_fixed, AMASS}

    # data representation used
    dataset_type = '3D'  # {3D, Euler}

    # type of labels used
    labels_type = 'clustered' #{For CMU: clustered, zero, For Human3.6: simplified, clustered, naive_clustered, zero}
    
    labels_type_ablation = None #{nogan, onlygan}

    experiment_name = join(dataset_name,
                     dataset_type ,
                     join(labels_type,
                     join('h' + str(hidden_units),
                     'n_in' + str(n_in),
                     'n_out' + str(n_out))))

    if loss_decay:
        experiment_name = join(experiment_name, 'loss_decay'+ str(decay_factor))

    # for ablation study
    warmup_frames_exp = False
    if warmup_frames_exp:
        warmup_frames = 0
        warmup_frames = 1
        warmup_frames = 3
        # warmup_frames = 10
        experiment_name = join(experiment_name, 'warm_up_exp_'+str(warmup_frames))

    # number of clusters ablation study
    number_of_clusteres_ablation = False
    if number_of_clusteres_ablation:
        label_dim = 4
        experiment_name = join(experiment_name, 'label_dim_exp'+str(label_dim))

    wo_disc = False
    if wo_disc:
        experiment_name = join(experiment_name, 'wo_disc')

    #Visualize
    epoch_to_visualize = str(99)
    auto_regress_vis = 7
        
    parent_path = dirname(abspath(__file__))
    parent_path = abspath(dirname(abspath(__file__)) + "/../")
    output_path = abspath(join(parent_path, 'output'))+'/'

    vis_root = join(output_path, 'visualize', experiment_name , epoch_to_visualize)
    file_root = join(output_path, 'files', experiment_name , epoch_to_visualize)

    random_seed = 1234567890

    long_term_inference = True
    long_term_auto_regress_vis = 13
    # long_term_auto_regress_vis = 38
    num_seeds = 8

    if dataset_name == 'H36M' and dataset_type == '3D':
        data_dim = 96
        num_seeds = 256
        num_seeds = 8
    if dataset_name == 'H36M' and dataset_type == 'Euler':
        data_dim = 99
        num_seeds = 8
    if dataset_name == 'H36M_less_joints' and dataset_type == '3D':
        data_dim = 51
        num_seeds = 256
    if dataset_name == 'H36M_less_joints_fixed' and dataset_type == '3D':
        data_dim = 51
        num_seeds = 256
    if dataset_name == 'CMU' and dataset_type == '3D':
        data_dim = 114
        num_seeds = 8
        framrates = [30]
    if dataset_name == 'CMU' and dataset_type == 'Euler':
        data_dim = 117
        num_seeds = 8
        framrates = [30]
    if dataset_name == 'AMASS' and dataset_type == '3D':
        data_dim = 66
        num_seeds = 8
        framrates = [30]
        
    # if videos are to be generated
    plot_videos = False #{True, False}

    frames_to_visualize = 100

    loss_ablation_experiment = False
    if loss_ablation_experiment:
        loss_ablation_exp = 1 # Only reconstruction loss
        # loss_ablation_exp = 2 # Only GAN loss

        vis_root = join(output_path,'visualize', experiment_name, 'loss_ablation_experiment', str(loss_ablation_exp))
        file_root = join(output_path, 'files', experiment_name, 'loss_ablation_experiment', str(loss_ablation_exp))
        experiment_name = join(experiment_name, 'loss_ablation_exp' + str(loss_ablation_exp))

    #noise experiment, for single input, the code produces multiple output
    noise_experiment = False
    number_of_noises = 10

    if noise_experiment:
        vis_root = join(output_path, 'visualize_noise', experiment_name, epoch_to_visualize)
        file_root = join(output_path, 'files_noise', experiment_name, epoch_to_visualize)

        # Only with labels type ablation
        # vis_root = join(vis_root, labels_type_ablation)
        # file_root = join(file_root, labels_type_ablation)

        if dataset_name == 'H36M' and dataset_type == '3D':
            data_dim = 96
            num_seeds = 8
        if dataset_name == 'H36M_less_joints' and dataset_type == '3D':
            data_dim = 51
            num_seeds = 8
        if dataset_name == 'H36M_less_joints_fixed' and dataset_type == '3D':
            data_dim = 51
            num_seeds = 8
        if dataset_name == 'CMU' and dataset_type == 'Euler':
            data_dim = 117
            num_seeds = 8
            framrates = [30]
            
    # noise factor by which noise is multiplied
    noise_factor = 5 #{0, 1, 5}

    #forecast label experiment
    forecast_experiment = False

    if forecast_experiment:
        epoch_to_visualize = str(99)
        #number of forecast frames for current pose
        forecast = 1  #{20, 5, 2, 1}

        vis_root = join('output_path visualize', experiment_name, 'forecast_experiment', str(forecast))
        file_root = join('output_path files', experiment_name, 'forecast_experiment', str(forecast))
        experiment_name = join(experiment_name, 'forecast_experiment' + str(forecast))
        auto_regress_vis = 4

    #eulicdean loss upto frames experiment
    eulicdean_loss_upto_frames_experiment = False

    if eulicdean_loss_upto_frames_experiment:
        loss_after_frames_exp = 1 #decay to "0"
        loss_after_frames_exp = 2 #decay to "0.5"
        loss_after_frames_exp = 3 #no decay

        vis_root = join('output_path visualize', experiment_name, 'loss_after_frames_exp', str(loss_after_frames_exp))
        file_root = join('output_path files', experiment_name, 'loss_after_frames_exp', str(loss_after_frames_exp))
        experiment_name = join(experiment_name, 'lossafter_frames_experiment' + str(loss_after_frames_exp))
        auto_regress_vis = 6

    # User study
    user_study = False

    if user_study:
        auto_regress_vis = 1
        frames_to_visualize = 25
        # #
        auto_regress_vis = 2
        frames_to_visualize = 50

        auto_regress_vis = 4
        frames_to_visualize = 100

        # auto_regress_vis = 14
        # frames_to_visualize = 300

        vis_root = join(output_path, 'visualize_user_study', experiment_name, 'userstudy', str(frames_to_visualize))
        file_root = join(output_path, 'files_user_study', experiment_name, 'userstudy', str(frames_to_visualize))

    #vis_clusters, input, the clusters are visualized 
    vis_clusters = False

    if vis_clusters:
        plot_videos = True #{True, False}
        vis_root = join(output_path, 'visualize_labels', experiment_name, epoch_to_visualize)
        file_root = join(output_path, 'files_labels', experiment_name, epoch_to_visualize)

        if dataset_name == 'H36M_less_joints' and dataset_type == '3D':
            data_dim = 51
            num_seeds = 8