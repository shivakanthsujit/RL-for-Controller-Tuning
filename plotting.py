import os
import glob
import pickle
import matplotlib.pyplot as plt
import pprint
import csv
import json
import numpy as np
import PIL
import PIL.Image as Image

seeds = np.arange(6)
seeds = [0]
for seed in seeds:
    files = glob.glob(f'logs/**/{seed}/test_files/**/env_obj*.pkl', recursive=True)
    envs = ['CSTR2PID']
    methods=['PPO', 'DDPG']
    methods_plot=['PPO']
    plot_types = ['servo', 'regulatory', 'servo-noise', 'regulatory-noise']
    plot_types = ['servo-regulatory-stable', 'servo-regulatory-unstable', 'servo-regulatory-stable-noise', 'servo-regulatory-unstable-noise']
    plot_types = ['servo-stable']
    # plot_types = ['servo-regulatory-stable']
    # plot_types = ['parameter-uncertainity']
    plot_names = {k: k.title().replace('-', ' ') for k in plot_types}
    type_files = {env: {k: [f for f in files if f.split('/')[-2] == k and env in f and not('ablation' in f)] for k in plot_types} for env in envs}
    pprint.pprint(type_files)
    legend_colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    headers = ['name', 'ise', 'iae']
    plots_dir = 'plots_servoregulatory_new_2'
    data = {}
    for env_name in envs:
        metrics = {plot_type: [] for plot_type in plot_types}
        save_dir = os.path.join(plots_dir, f'{env_name}')
        os.makedirs(save_dir, exist_ok=True)
        for plot_type in plot_types:
            data[plot_type] = {'Y': {}, 'U': {}, 'axis': {}, 'input': {}}
            print("Plot Type: ", plot_type)
            if len(type_files[env_name][plot_type]) > 0:
                csv_folder = os.path.join(save_dir, 'values')
                os.makedirs(csv_folder, exist_ok=True)
                for i, plot in enumerate(type_files[env_name][plot_type]):
                    with open(plot, 'rb') as f:
                        tag_name = plot.split('/')[2].split('_')[0]
                        if tag_name in methods:
                            env = pickle.load(f)
                            if i == 0:
                                data[plot_type]['Y']['Setpoint'] = env.r[: env.k]
                                data[plot_type]['axis']['axis'], data[plot_type]['axis']['axis_name'] = env.get_axis()
                            data[plot_type]['Y'][tag_name] = env.y[: env.k]
                            data[plot_type]['U'][tag_name] = env.u[: env.k]
                            data[plot_type]['input'][tag_name] = env.input[: env.k]

                            # eval_img = env.plot_gains(True)
                            # plt.close()
                            # im = Image.fromarray(eval_img)
                            # fname = os.path.join(save_dir, f'{plot_type}_{seed}_{tag_name}_gains.png')
                            # im.save(fname)

                            ise = env.ise()
                            iae = env.iae()
                            piecewise_ise = env.get_piecewise_ise()
                            piecewise_iae = env.get_piecewise_iae()
                            metrics[plot_type].append([tag_name, ise, iae] + piecewise_ise + piecewise_iae)


                            # csv_path = os.path.join(csv_folder, f'{plot_type}.csv')
                            # with open(csv_path, 'w') as csvf:
                            #     write = csv.writer(csvf)
                            #     csv_headers = ['Setpoint', 'Y', 'U']
                            #     write.writerow(csv_headers)
                            #     csv_data = np.array([data[plot_type]['Y'][tag_name], data[plot_type]['Y'][tag_name], data[plot_type]['U'][tag_name]]).transpose()
                            #     write.writerows(csv_data)
                save_f = os.path.join(save_dir, plot_type)
                os.makedirs(save_f, exist_ok=True)
                with open(os.path.join(save_f, f'metrics_{seed}.csv'), 'w') as csvf:
                    write = csv.writer(csvf)
                    csv_headers = headers + [f'ise_{env.indices[i]}_{env.indices[i+1]}' for i in range(len(piecewise_ise))] + [f'iae_{env.indices[i]}_{env.indices[i+1]}' for i in range(len(piecewise_iae))]
                    write.writerow(csv_headers)
                    write.writerows(metrics[plot_type])

                fig = plt.figure(figsize=(12, 7))
                fig.suptitle(f'{plot_names[plot_type]}')
                for i, (k, v) in enumerate(data[plot_type]['Y'].items()):
                    if k == 'Setpoint':
                        plt.step(data[plot_type]['axis']['axis'], v, label=k, linestyle="dashed", where="post", color=legend_colours[i], alpha=0.5,linewidth=2.0)
                    elif k in methods_plot:
                        plt.plot(data[plot_type]['axis']['axis'], v, label=f"{k}-PID", color=legend_colours[i], linewidth=2.0)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, ncol=3)
                plt.ylabel('Reactor Temperature K')
                plt.xlabel(data[plot_type]['axis']['axis_name'])
                plt.xlim(data[plot_type]['axis']['axis'][0], data[plot_type]['axis']['axis'][-1])
                plt.grid()

                plt.tight_layout()
                fname = os.path.join(save_f, f'output_{seed}.png')
                plt.savefig(fname, bbox_inches='tight')
                plt.close()

                fig = plt.figure(figsize=(12, 7))
                fig.suptitle(f'{plot_names[plot_type]}')
                for i, (k, v) in enumerate(data[plot_type]['U'].items()):
                    if k in methods_plot:
                        plt.plot(data[plot_type]['axis']['axis'], v, label=f"{k}-PID", color=legend_colours[i+1], linewidth=2.0)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, ncol=2)
                plt.ylabel('Jacket Temperature K')
                plt.xlabel(data[plot_type]['axis']['axis_name'])
                plt.xlim(data[plot_type]['axis']['axis'][0], data[plot_type]['axis']['axis'][-1])
                plt.grid()

                plt.tight_layout()
                fname = os.path.join(save_f, f'u_{seed}.png')
                plt.savefig(fname, bbox_inches='tight')
                plt.close()

                for i, (k, v) in enumerate(data[plot_type]['input'].items()):
                    if k in methods_plot:
                        fig = plt.figure(figsize=(12, 7))
                        fig.suptitle(f'{plot_names[plot_type]}')
                        lineObj = plt.plot(data[plot_type]['axis']['axis'], v, linewidth=2.0)
                        plt.legend(lineObj, ['Kp', 'Ki', 'Kd'], loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, ncol=2)
                        plt.ylabel('Jacket Temperature K')
                        plt.xlabel(data[plot_type]['axis']['axis_name'])
                        plt.xlim(data[plot_type]['axis']['axis'][0], data[plot_type]['axis']['axis'][-1])
                        plt.grid()

                        plt.tight_layout()
                        fname = os.path.join(save_f, f'gains_{seed}_{k}.png')
                        plt.savefig(fname, bbox_inches='tight')
                        plt.close()

        json_fname = os.path.join(save_dir, 'metrics.json')
        with open(json_fname, 'w') as f:
            json.dump(metrics, f)