
function result = load_data(data_dir_path, output_dir_path)
    data_files = dir(fullfile(data_dir_path, '*.gdf'));

    for k = 1:length(data_files)
        file_path = fullfile(data_files(k).folder, data_files(k).name);
        disp(['Processing file: ', file_path]);

        [signal, header] = sload(file_path);
        event_position = header.EVENT.POS;
        event_type = header.EVENT.TYP;
        event_duration = header.EVENT.DUR;

        [folder, file_name, ext] = fileparts(file_path);

        eeg_data_output_file_path = fullfile(output_dir_path, ...
                                    [file_name '_eeg_data' '.csv']);

        event_position_output_file_path = fullfile(output_dir_path, ...
                                    [file_name '_event_position' '.csv']);

        event_type_output_file_path = fullfile(output_dir_path, ...
                                    [file_name '_event_type' '.csv']);

        event_duration_output_file_path = fullfile(output_dir_path, ...
                                    [file_name '_event_duration' '.csv']);

        writematrix(signal, eeg_data_output_file_path);
        writematrix(event_position, event_position_output_file_path);
        writematrix(event_type, event_type_output_file_path);
        writematrix(event_duration, event_duration_output_file_path);
    end

    result = 0;
end

root_dir_path = 'E:\data\eeg\bci_competition';
data_dir_path = fullfile(root_dir_path, 'BCICIV_2a_gdf');
output_dir_path = fullfile(root_dir_path, 'BCICIV_2a_csv');

result = load_data(data_dir_path, output_dir_path);

root_dir_path = 'E:\data\eeg\bci_competition';
data_dir_path = fullfile(root_dir_path, 'BCICIV_2b_gdf');
output_dir_path = fullfile(root_dir_path, 'BCICIV_2b_csv');

result = load_data(data_dir_path, output_dir_path);

root_dir_path = 'E:\data\eeg\bci_competition';
label_dir_path = fullfile(root_dir_path, 'BCICIV_2a_labels', 'true_labels');
output_dir_path = fullfile(root_dir_path, 'BCICIV_2a_csv');

root_dir_path = 'E:\data\eeg\bci_competition';
label_dir_path = fullfile(root_dir_path, 'BCICIV_2b_labels', 'true_labels');
output_dir_path = fullfile(root_dir_path, 'BCICIV_2b_csv');


function result = load_data(label_dir_path, output_dir_path)
    label_files = dir(fullfile(label_dir_path, '*.mat'));

    for k = 1:length(label_files)
        file_path = fullfile(label_files(k).folder, label_files(k).name);
        disp(['Processing file: ', file_path]);

        label_info = load(file_path);
        labels = label_info.classlabel;

        [folder, file_name, ext] = fileparts(file_path);

        labels_output_file_path = fullfile(output_dir_path, ...
                                    [file_name '_labels' '.csv']);

        writematrix(labels, labels_output_file_path);
    end

    result = 0;
end

subject_index = 6; % 1-9
%% T data
session_type = 'T'; % T and E
dir_1 = ['E:\data\eeg\bci_competition\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf']; % set your path of the downloaded data
[s, HDR] = sload(dir_1);
% Label
% label = HDR.Classlabel;
labeldir_1 = ['E:\data\eeg\bci_competition\BCICIV_2a_labels\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir_1);
label_1 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS; % use POS to get trials
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_1 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_1(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_1(isnan(data_1)) = 0;

% E data
session_type = 'E';
dir_2 = ['E:\data\bci_competition\BCICIV_2a_gdf\A0',num2str(subject_index),session_type,'.gdf'];
% dir = 'D:\Lab\MI\BCICIV_2a_gdf\A01E.gdf';
[s, HDR] = sload(dir_2);

% Label
% label = HDR.Classlabel;
labeldir_2 = ['E:\data\bci_competition\BCICIV_2a_labels\true_labels\A0',num2str(subject_index),session_type,'.mat'];
load(labeldir_2);
label_2 = classlabel;

% construct sample - data Section 1000*22*288
Pos = HDR.EVENT.POS;
% Dur = HDR.EVENT.DUR;
Typ = HDR.EVENT.TYP;

k = 0;
data_2 = zeros(1000,22,288);
for j = 1:length(Typ)
    if  Typ(j) == 768
        k = k+1;
        data_2(:,:,k) = s((Pos(j)+500):(Pos(j)+1499),1:22);
    end
end

% wipe off NaN
data_2(isnan(data_2)) = 0;

%% preprocessing
% option - band-pass filter
fc = 250; % sampling rate
Wl = 4; Wh = 40; % pass band
Wn = [Wl*2 Wh*2]/fc;
[b,a]=cheby2(6,60,Wn);
for j = 1:288
    data_1(:,:,j) = filtfilt(b,a,data_1(:,:,j));
    data_2(:,:,j) = filtfilt(b,a,data_2(:,:,j));
end

% option - a simple standardization
%{
eeg_mean = mean(data,3);
eeg_std = std(data,1,3);
fb_data = (data-eeg_mean)./eeg_std;
%}

%% Save the data to a mat file
data = data_1;
label = label_1;
% label = t_label + 1;
saveDir = ['E:\data\bci_competition\standard_2a_data\A0',num2str(subject_index),'T.mat'];
save(saveDir,'data','label');

data = data_2;
label = label_2;
saveDir = ['E:\data\bci_competition\standard_2a_data\A0',num2str(subject_index),'E.mat'];
save(saveDir,'data','label');
