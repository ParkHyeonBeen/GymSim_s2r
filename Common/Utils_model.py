from Common.Utils import *

def save_model(network, loss_best, loss_now, path, ard=False):
    if loss_best > loss_now:
        if ard:
            torch.save(network.state_dict(), path + "/best_" + path[-3:])
        else:
            torch.save(network.state_dict(), path + "/better_" + path[-3:])
        return loss_now
    else:
        if not ard:
            torch.save(network.state_dict(), path + "/current_" + path[-3:])

def load_models(args_tester, model):

    path = args_tester.path
    path_model = None
    path_invmodel = None

    if "DNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_fname + "saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_fname + "saved_net/model/DNN/inv" + args_tester.modelnet_name

    if "BNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_fname + "saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_fname + "saved_net/model/BNN/inv" + args_tester.modelnet_name

    if args_tester.develop_mode == "MRAP":
        model.load_state_dict(torch.load(path_model))
    if args_tester.develop_mode == "DeepDOB":
        if "bnn" in args_tester.result_fname:
            model_tmp = torch.load(path_invmodel)
            for key in model_tmp.copy().keys():
                if 'eps' in key:
                    del(model_tmp[key])
            model.load_state_dict(model_tmp)
        else:
            model.load_state_dict(torch.load(path_invmodel))

def validate_measure(error_list):
    error_max = np.max(error_list, axis=0)
    mean = np.mean(error_list, axis=0)
    std = np.std(error_list, axis=0)
    loss = np.sqrt(mean**2 + std**2)

    return [loss, mean, std, error_max]

def get_random_action_batch(observation, env_action, test_env, model_buffer, max_action, min_action):

    env_action_noise, _ = add_noise(env_action, scale=0.1)
    action_noise = normalize(env_action_noise, max_action, min_action)
    next_observation, reward, done, info = test_env.step(env_action_noise)
    model_buffer.add(observation, action_noise, reward, next_observation, float(done))

def set_sync_env(env, test_env):

    position = env.sim.data.qpos.flat.copy()
    velocity = env.sim.data.qvel.flat.copy()

    test_env.set_state(position, velocity)