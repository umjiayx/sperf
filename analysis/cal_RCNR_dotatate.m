clear
clc

patient_ids_all = {'patient1_c1_s2', 'patient2_c1_s4', 'patient3_c2_s4', ...
    'patient4_c1_s4', 'patient5_c1_s4', 'patient6_c1_s4', 'patient7_c1_s4', ...
    'patient8_c1_s4', 'patient9_c2_s2', 'patient10_c1_s4', 'patient11_c1_s4'};

bkg_file_names = {...
    'spleen_processed.mat' % 1
    };

patients_offsets = get_patient_offsets;
ITER = 16;

bkg_idx = ones(1, length(patient_ids_all)) * 6;

RCNRTableFilename = 'RCNR_table_d=2.mat';

RCNR_Lesion_sperf = [];
RCNR_Kidney_sperf = [];
RCNR_Bone_Marrow_sperf = [];

RCNR_Lesion_partial = [];
RCNR_Kidney_partial = [];
RCNR_Bone_Marrow_partial = [];

RCNR_Lesion_linear = [];
RCNR_Kidney_linear = [];
RCNR_Bone_Marrow_linear = [];    

for idx = 1:length(patient_ids_all)
    patient_id = patient_ids_all{idx};
    save_RCNR_dir = './RCNR/';

    if ~exist(save_RCNR_dir, "dir")
        mkdir(save_RCNR_dir)
    end
    
    roi_mask_folder = strcat('./masks/', patient_id, '/ROI/');
    mask_dirs = dir(roi_mask_folder);
    
    masks_path = dir(fullfile(roi_mask_folder, '*_processed.mat'));
    mask_file_names = {masks_path.name};
    dotFiles = startsWith(mask_file_names, '._');
    mask_file_names(dotFiles) = [];
    mask_file_names = unique(mask_file_names);
    
    % background
    bkg_path = strcat('./masks/', patient_id, '/BKG/', bkg_file_names{bkg_idx(idx)});
    bkg_mask = load(bkg_path).mask;    

    sperfrecon_path = strcat('./recon_d=2/sperfrecon/', patient_id, '/sperfrecon_81_20iters.fld');
    fullrecon_path = strcat('./recon_d=2/fullrecon/', patient_id, '/fullrecon_81_20iters.fld');
    partialrecon_path = strcat('./recon_d=2/partialrecon/', patient_id, '/partialrecon_81_20iters.fld');
    linearrecon_path = strcat('./recon_d=2/linearrecon/', patient_id, '/linearrecon_81_20iters.fld');

    sperfrecon = fld_read(sperfrecon_path);
    fullrecon = fld_read(fullrecon_path);
    partialrecon = fld_read(partialrecon_path);
    linearrecon = fld_read(linearrecon_path);

    s = sperfrecon(:,:,:,ITER);
    f = fullrecon(:,:,:,ITER);
    p = partialrecon(:,:,:,ITER);
    l = linearrecon(:,:,:,ITER);

    X_offset = patients_offsets(idx).X;
    Y_offset = patients_offsets(idx).Y;
    Z_offset = patients_offsets(idx).Z;

    s = interpto512_DOTATATE_lu177(s, X_offset, Y_offset, Z_offset);
    f = interpto512_DOTATATE_lu177(f, X_offset, Y_offset, Z_offset);
    p = interpto512_DOTATATE_lu177(p, X_offset, Y_offset, Z_offset);
    l = interpto512_DOTATATE_lu177(l, X_offset, Y_offset, Z_offset);    

    fprintf('----- Calculating RCNR for %s -----\n', patient_id);

    for mask_id = 1:length(mask_file_names)
        mask_file_name = mask_file_names{mask_id};
        mask_file_path = strcat('./masks/', patient_id, '/ROI/', mask_file_name);
        mask = load(mask_file_path).mask;

        % f
        mean_lesion = mean(f(mask==1));
        mean_bkg = mean(f(bkg_mask==1));
        std_bkg = std(f(bkg_mask==1));
        CNR_f = (mean_lesion - mean_bkg) / std_bkg;
        
        % s
        mean_lesion = mean(s(mask==1));
        mean_bkg = mean(s(bkg_mask==1));
        std_bkg = std(s(bkg_mask==1));
        CNR_s = (mean_lesion - mean_bkg) / std_bkg;        

        % l
        mean_lesion = mean(l(mask==1));
        mean_bkg = mean(l(bkg_mask==1));
        std_bkg = std(l(bkg_mask==1));
        CNR_l = (mean_lesion - mean_bkg) / std_bkg;
        
        % p
        mean_lesion = mean(p(mask==1));
        mean_bkg = mean(p(bkg_mask==1));
        std_bkg = std(p(bkg_mask==1));
        CNR_p = (mean_lesion - mean_bkg) / std_bkg;       

        RCNR_s = CNR_s / CNR_f;
        RCNR_l = CNR_l / CNR_f;
        RCNR_p = CNR_p / CNR_f;

        if contains(mask_file_name, 'lesion')

            RCNR_Lesion_sperf(end+1) = RCNR_s;
            RCNR_Lesion_partial(end+1) = RCNR_p;
            RCNR_Lesion_linear(end+1) = RCNR_l;                           

        elseif contains(mask_file_name, 'KIDNEY')

            RCNR_Kidney_sperf(end+1) = RCNR_s;
            RCNR_Kidney_partial(end+1) = RCNR_p;
            RCNR_Kidney_linear(end+1) = RCNR_l;

        else
            fprintf("Something unexpected happend: %s!\n", mask_file_path);
            
        end

    end

end
%%
% Calculate the means for each region
Lesion_Means = [mean(RCNR_Lesion_sperf), mean(RCNR_Lesion_partial), mean(RCNR_Lesion_linear)];
Kidney_Means = [mean(RCNR_Kidney_sperf), mean(RCNR_Kidney_partial), mean(RCNR_Kidney_linear)];

% Combine all the data into a matrix
DataMatrix = [Lesion_Means; Kidney_Means];

% Define updated row titles
RowTitles = {'Lesions', 'Kidneys'};
ColumnTitles = {'SpeRF', 'Partial', 'LinInt'};

% Create the updated table
RCNRTable = array2table(DataMatrix, 'VariableNames', ColumnTitles, 'RowNames', RowTitles);

fprintf('\n') 

% Display the table in the MATLAB console
disp(RCNRTable);

% Save the table to a .mat file
resultsPath = fullfile(save_RCNR_dir, RCNRTableFilename);
save(resultsPath, 'RCNRTable', ...
    'RCNR_Lesion_sperf', 'RCNR_Kidney_sperf', ...
    'RCNR_Lesion_partial', 'RCNR_Kidney_partial', ...
    'RCNR_Lesion_linear', 'RCNR_Kidney_linear');

% Optionally, save as a .csv file for easy readability
%csvFilename = fullfile(save_RCNR_dir, strcat('RCNR_table_', currentDateTime, '.csv'));
%writetable(RCNRTable, csvFilename, 'WriteRowNames', true);

fprintf('Table saved to %s\n', fullfile(save_RCNR_dir, RCNRTableFilename));
