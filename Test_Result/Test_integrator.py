import os, sys, argparse
import Tester
from Tester import hyperparameters

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Common.Utils import *

Tester_data = DataManager()

def tester_hyperparameters():
    parser = argparse.ArgumentParser(description='Intergrated Testor')

    parser.add_argument('--result-fname', '-fn', default='hopper_sac', type=str, help='Result name to check')
    parser.add_argument('--develop-mode', '-dm', default='MRAP', help="Basic, DeepDOB, MRAP")
    parser.add_argument('--num-test', '-tn', default=100, type=int, help='the number of tests')
    parser.add_argument('--policy-kind', '-pk', default='best', type=str, help='current, better, best')
    parser.add_argument('--model-kind', '-mk', default='better', type=str, help='current, better')
    parser.add_argument('--model-order', '-mo', default=['DNN'], help='the order of test model')

    parser.add_argument('--dist-kind', '-dk', default='sine', type=str, help='normal, sine, sine_normal')
    parser.add_argument('--num-dist', '-dn', default=20, type=int, help='the number of disturbance in certain range')
    parser.add_argument('--add-to', '-ad', default='action', type=str, help='action, state')
    parser.add_argument('--max-dist-action', '-xda', default=1.0, type=float, help='max mag of dist for action')
    parser.add_argument('--min-dist-action', '-nda', default=0.0, type=float, help='min mag of dist for action')
    parser.add_argument('--max-dist-state', '-xds', default=0.5, type=float, help='max mag of dist for state')
    parser.add_argument('--min-dist-state', '-nds', default=0.15, type=float, help='min mag of dist for state')

    parser.add_argument('--save-dir', default="/home/phb/SBPO/Results/", help='path of saved data')

    args_itg = parser.parse_known_args()

    return args_itg


args_itg = tester_hyperparameters()[0]
model_list = ["dnn_" + args_itg.model_kind for name in args_itg.model_order]

if args_itg.add_to == 'state':
    start_mag = args_itg.min_dist_state
    mag_range = args_itg.max_dist_state
else:
    start_mag = args_itg.min_dist_action
    mag_range = args_itg.max_dist_action

print("result name:", args_itg.result_fname)
print("action range: +-1")
print("the kind of external force:", args_itg.dist_kind)
print("the number of test:", args_itg.num_test)
print("the number of external force:", args_itg.num_dist)

print("start time : %s" % time.strftime("%Y%m%d-%H%M%S"))
start_time = time.time()

if args_itg.develop_mode == 'Basic':
    print("start Basic PG algorithm")
    Tester_data.init_data()
    for i in np.linspace(start_mag, mag_range, args_itg.num_dist+1):
        print("current external force scale :", i)
        if args_itg.dist_kind == 'sine':
            args = hyperparameters(result_fname=args_itg.result_fname,
                                   num_test=args_itg.num_test,
                                   disturbance_scale=i,
                                   add_to=args_itg.add_to,
                                   policy_name="policy_"+args_itg.policy_kind
                                   )
        else:
            args = hyperparameters(result_fname=args_itg.result_fname,
                                   num_test=args_itg.num_test,
                                   noise_scale=i,
                                   add_to=args_itg.add_to,
                                   policy_name="policy_"+args_itg.policy_kind
                                   )
        reward_avg, reward_max, reward_min, reward_std, alive_rate = Tester.main(args)
        saveData = np.array([reward_avg, reward_max, reward_min, reward_std, alive_rate])
        Tester_data.put_data(saveData)
    Tester_data.save_data(args_itg.save_dir, time.strftime("%m%d_") + args_itg.result_fname + '_' + args_itg.develop_mode + '_' + args_itg.dist_kind+'2'+args_itg.add_to)
    print("finish time of Basic: %s" % time.strftime("%Y%m%d-%H%M%S"))
    print("elapsed time : ", time.time() - start_time)
else:
    print("start DeepDOB algorithm")
    for model in model_list:
        print("start time of "+model+": %s" % time.strftime("%Y%m%d-%H%M%S"))
        start_time = time.time()
        print("The model to test :", model)
        Tester_data.init_data()
        for i in np.linspace(start_mag, mag_range, args_itg.num_dist+1):
            print("current external force scale :", i)
            if args_itg.dist_kind == 'sine':
                args = hyperparameters(result_fname=args_itg.result_fname,
                                       num_test=args_itg.num_test,
                                       disturbance_scale=i,
                                       add_to=args_itg.add_to,
                                       policy_name="policy_"+args_itg.policy_kind,
                                       model_name=model
                                       )
            else:
                args = hyperparameters(result_fname=args_itg.result_fname,
                                       num_test=args_itg.num_test,
                                       noise_scale=i,
                                       add_to=args_itg.add_to,
                                       policy_name="policy_"+args_itg.policy_kind,
                                       model_name=model
                                       )
            reward_avg, reward_max, reward_min, reward_std, alive_rate = Tester.main(args)
            saveData = np.array([reward_avg, reward_std])
            Tester_data.put_data(saveData)
        Tester_data.plot_variance_fig(args_itg.save_dir + args_itg.result_fname + "/saved_log/" + time.strftime("%m%d_") + '_' + model + '_' + args_itg.dist_kind+'2'+args_itg.add_to)
        print("finish time of" + model + ": %s" % time.strftime("%Y%m%d-%H%M%S"))
        print("elapsed time : ", time.time() - start_time)