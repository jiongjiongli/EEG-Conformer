
root_dir_path = 'E:\data\eeg\bci_competition';
data_dir_path = fullfile(root_dir_path, 'BCICIV_2a_gdf');
label_dir_path = fullfile(root_dir_path, 'BCICIV_2a_labels', 'true_labels');
output_dir_path = fullfile(root_dir_path, 'BCICIV_2a_mat');

result = preprocess_data(data_dir_path, label_dir_path, output_dir_path);
fprintf('Completed!\n')


function result = preprocess_data(data_dir_path, label_dir_path, output_dir_path)
    num_samples = 288;
    seq_len = 1000;
    num_channels = 22;
    sample_freq = 250;
    % high-pass (lower cutoff)
    l_freq = 4;
    % low-pass (upper cutoff)
    h_freq = 40;

    Wn = [l_freq * 2, h_freq * 2] / sample_freq;
    [b, a]=cheby2(6, 60, Wn);

    data_files = dir(fullfile(data_dir_path, '*.gdf'));

    for k = 1:length(data_files)
        file_path = fullfile(data_files(k).folder, data_files(k).name);
        disp(['Processing file: ', file_path]);

        [signal, header] = sload(file_path);
        event_position = header.EVENT.POS;
        event_type = header.EVENT.TYP;
        event_duration = header.EVENT.DUR;

        eeg_data_idx = 0;
        eeg_data = zeros(seq_len, num_channels, num_samples);
        for j = 1:length(event_type)
            % 768 : "Start of a trial"
            if event_type(j) == 768
                eeg_data_idx = eeg_data_idx + 1;
                event_start = event_position(j) + sample_freq * 2;
                event_end = event_position(j) + sample_freq * 6 - 1;
                eeg_data(:, :, eeg_data_idx) = signal(event_start:event_end, 1:num_channels);
            end
        end

        % wipe off NaN
        eeg_data(isnan(eeg_data)) = 0;

        for eeg_data_idx = 1:num_samples
            eeg_data(:,:,eeg_data_idx) = filtfilt(b, a, eeg_data(:,:,eeg_data_idx));
        end

        [folder, file_name, ext] = fileparts(file_path);
        label_file_path = fullfile(label_dir_path, [file_name, '.mat']);
        label_info = load(label_file_path);
        labels = label_info.classlabel;

        output_file_path = fullfile(output_dir_path, ...
                                    [file_name, '.mat']);
        save(output_file_path,'eeg_data', 'labels');
    end

    result = 0;
end
