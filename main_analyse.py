import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pattern import Pattern


def create_guess_list_files(device, model, num_group, in_dirs, out_list_path_format):
    for i in range(num_group):
        open(out_list_path_format.format(i+1), "w").close()
    for in_dir in in_dirs:
        for root, dir_names, file_names in os.walk(in_dir):
            for i, file_name in enumerate(file_names):
                data_input_nps = np.random.permutation(list(np.load(os.path.join(root, file_name)).values())[0])
                for data_input_np in data_input_nps:
                    data_input = torch.from_numpy(data_input_np).float()
                    output = model(data_input.to(device))
                    pcgnum = torch.max(output, 0)[1].item() + 1  # predicted with the most confidence
                    with open(out_list_path_format.format(pcgnum), "a") as file_out:
                        file_out.write(os.path.join(root, file_name) + "\n")
                    print(f"\r\tcreate guess list: {i}/{len(file_names)}", end="")
            print(f"\rcreate guess list: {len(file_names)}")


def get_counts(guess_list_path_format):
    confusion = np.zeros((17, 17)).astype(int)
    for i in range(17):
        with open(guess_list_path_format.format(i+1), "r") as list_file:
            for file_name in list_file:
                group_name = re.split(r"[_./]", file_name)[-2]
                group_number = Pattern.group_numbers[group_name]
                confusion[i, group_number - 1] += 1  # (guess, actual)

    confusion = confusion/confusion.sum(0)[0]
    confusion = confusion
    plt.figure(figsize=(10, 10))
    plt.gca().matshow(confusion, cmap="cividis")
    for i in range(17):
        for j in range(17):
            c = confusion[i, j]
            plt.gca().text(j, i, f"{c:.2f}", va='center', ha='center', color="grey")
    plt.gca().set_ylabel("Guess")
    plt.gca().xaxis.set_label_position('top')
    plt.gca().set_xlabel("Actual")
    plt.xticks(range(17), Pattern.group_names)
    plt.yticks(range(17), Pattern.group_names)
    plt.show()
    return confusion


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.LeakyReLU(),
        torch.nn.Linear(500, 250),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(250, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 17),
        torch.nn.LeakyReLU(),
    )
    state_dict = torch.load("state_dicts/state_dict_9_cluster_24")
    model.load_state_dict(state_dict["model"])
    model.eval()
    model = model.to(device)

    create_guess_list_files(
        device, model, num_group=17,
        in_dirs=["data/cluster_2/"],
        out_list_path_format="guess/plane_group_list_{}.txt"
    )

    get_counts("guess/plane_group_list_{}.txt")


if __name__ == '__main__':
    main()
