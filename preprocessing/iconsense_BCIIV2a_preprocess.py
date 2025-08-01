from pathlib import Path
import re
import pandas as pd
import mne


sample_freq = 250
l_freq = 4.0  # high-pass (lower cutoff)
h_freq = 40.0  # low-pass (upper cutoff)
# Event types:
# 276   0x0114 Idling EEG (eyes open)
# 277   0x0115 Idling EEG (eyes closed)
# 768   0x0300 Start of a trial
# 769   0x0301 Cue onset left (class 1)
# 770   0x0302 Cue onset right (class 2)"
# 771   0x0303 Cue onset foot (class 3)"
# 772   0x0304 Cue onset tongue (class 4)"
# 783   0x030F Cue unknown"
# 1023  0x03FF Rejected trial"
# 1072  0x0430 Eye movements"
# 32766 0x7FFE Start of a new run"
event_type_mapping = {
    276:   "Idling EEG (eyes open)",
    277:   "Idling EEG (eyes closed)",
    768:   "Start of a trial",
    769:   "Cue onset left (class 1)",
    770:   "Cue onset right (class 2)",
    771:   "Cue onset foot (class 3)",
    772:   "Cue onset tongue (class 4)",
    783:   "Cue unknown",
    1023:  "Rejected trial",
    1072:  "Eye movements",
    32766: "Start of a new run",
}

root_dir_path = r"E:\data\eeg\bci_competition\BCICIV_2a_csv"
root_dir_path = Path(root_dir_path)

csv_file_paths = root_dir_path.glob("*.csv")
csv_file_paths = list(csv_file_paths)

csv_file_paths.sort()

data_file_infos = {}

for csv_file_path in csv_file_paths:
    file_name = csv_file_path.stem
    file_name_parts = file_name.split("_", maxsplit=1)

    assert len(file_name_parts) in [2], file_name_parts

    subject = file_name_parts[0][:-1]
    split_type = file_name_parts[0][-1]

    assert split_type in ["E", "T"], split_type

    if split_type == "E":
        split = "test"
    else:
        assert split_type == "T", split_type
        split = "trainval"

    data_file_infos.setdefault(subject, {})

    data_file_infos[subject].setdefault(split, {})

    split_data_file_infos = data_file_infos[subject][split]

    assert file_name_parts[1] in ["eeg_data", "event_position", "event_type", "event_duration", "labels"], file_name_parts

    assert file_name_parts[1] not in split_data_file_infos, file_name_parts
    split_data_file_infos[file_name_parts[1]] = csv_file_path

data_file_info_list = []

for subject, subj_data_file_infos in data_file_infos.items():
    for split, split_data_file_infos in subj_data_file_infos.items():
        data_info = dict(
            subject=subject,
            split=split,
        )

        for data_type in ["eeg_data", "event_position", "event_type", "event_duration", "labels"]:
            data_info[data_type]=split_data_file_infos[data_type]

        data_file_info_list.append(data_info)


for data_file_info in data_file_info_list:
    subject = data_file_info["subject"]
    split = data_file_info["split"]
    eeg_data_df = pd.read_csv(data_file_info["eeg_data"], header=None)
    event_position_df = pd.read_csv(data_file_info["event_position"], names=["event_position"])
    event_type_df = pd.read_csv(data_file_info["event_type"], names=["event_type"])
    event_duration_df = pd.read_csv(data_file_info["event_duration"], names=["event_duration"])
    labels_df = pd.read_csv(data_file_info["labels"], names=["label"])

    # print(subject,
    #       eeg_data_df.shape,
    #       event_position_df.shape,
    #       event_type_df.shape,
    #       event_duration_df.shape,
    #       labels_df.shape)
    event_type_df["event_description"] = event_type_df["event_type"].map(event_type_mapping)

    event_df = pd.concat([event_type_df, event_position_df, event_duration_df], axis=1)
    # 768:   "Start of a trial"
    event_positions = event_df[event_df["event_type"] == 768]["event_position"]

    assert len(event_positions) == len(labels_df), (len(event_positions), len(labels_df))

    eeg_seqs = []
    filtered_eeg_seqs = []

    for event_position in event_positions:
        event_start = event_position + sample_freq * 2
        event_end = event_position + sample_freq * 6
        eeg_seq = eeg_data_df.iloc[event_start:event_end, :22].to_numpy()
        eeg_seqs.append(eeg_seq)

        # Apply bandpass filter
        # eeg_seq Shape = [1K, 22]
        filtered_data = mne.filter.filter_data(eeg_seq.T,
                                               sfreq=sample_freq,
                                               l_freq=l_freq,
                                               h_freq=h_freq)
        filtered_eeg_seq = filtered_data.T
        # filtered_eeg_seq Shape = [1K, 22]
        filtered_eeg_seqs.append(filtered_eeg_seq)

    eeg_seqs_array = np.concatenate(eeg_seqs, axis=0)
    eeg_seqs_df = pd.DataFrame(eeg_seqs_array, columns=eeg_data_df.columns[:22])
    filtered_eeg_seqs_array = np.concatenate(filtered_eeg_seqs, axis=0)
    filtered_eeg_seqs_df = pd.DataFrame(filtered_eeg_seqs_array, columns=eeg_data_df.columns[:22])

    file_name_parts = data_file_info["eeg_data"].stem.split("_", maxsplit=1)
    file_name_prefix = file_name_parts[0]
    eeg_seqs_file_name = f"{file_name_prefix}_eeg_seqs.csv"
    eeg_seqs_file_path = data_file_info["eeg_data"].parent / eeg_seqs_file_name
    eeg_seqs_df.to_csv(eeg_seqs_file_path)

    filtered_eeg_seqs_file_name = f"{file_name_prefix}_filtered_eeg_seqs.csv"
    filtered_eeg_seqs_file_path = data_file_info["eeg_data"].parent / filtered_eeg_seqs_file_name
    filtered_eeg_seqs_df.to_csv(filtered_eeg_seqs_file_path)

print("Completed!")
