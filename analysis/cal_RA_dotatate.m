clear
clc

ITER = 16;

patient_ids_all = {
    'patient1_c1_s2', 'patient2_c1_s4', 'patient3_c2_s4', ...
    'patient4_c1_s4', 'patient5_c1_s4', 'patient6_c1_s4', 'patient7_c1_s4', ...
    'patient8_c1_s4', 'patient9_c2_s2', 'patient10_c1_s4', 'patient11_c1_s4'
};

patients_offsets = get_patient_offsets;

currentDateTime = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-SS'); %#ok<DATST>
RATableFilename = strcat('RA_table_', currentDateTime, '.mat');

RA_Lesion_sperf = [];
RA_Kidney_sperf = [];
RA_Bone_Marrow_sperf = [];
RA_Lesion_partial = [];
RA_Kidney_partial = [];
RA_Bone_Marrow_partial = [];
RA_Lesion_linear = [];
RA_Kidney_linear = [];
RA_Bone_Marrow_linear = [];    

RA_All_Organs_sperf = [];
RA_All_Organs_partial = [];
RA_All_Organs_linear = [];

%% 
for idx = 1:length(patient_ids_all)
    %% 
    patient_id = patient_ids_all{idx};
    save_ra_dir = strcat('./RA/');

    if ~exist(save_ra_dir, "dir")
        mkdir(save_ra_dir)
    end
    
    %% Mask file name
    roi_mask_folder = strcat('./masks/', patient_id, '/ROI/');
    mask_dirs = dir(roi_mask_folder);
    
    masks_path = dir(fullfile(roi_mask_folder, '*_processed.mat'));
    mask_file_names = {masks_path.name};
    dotFiles = startsWith(mask_file_names, '._');
    mask_file_names(dotFiles) = [];
    mask_file_names = unique(mask_file_names);    

    %% Read in files
    sperfrecon_path = strcat('./recon_d=8/sperfrecon/', patient_id, '/sperfrecon_81_20iters.fld');
    fullrecon_path = strcat('./recon_d=8/fullrecon/', patient_id, '/fullrecon_81_20iters.fld');
    partialrecon_path = strcat('./recon_d=8/partialrecon/', patient_id, '/partialrecon_81_20iters.fld');
    linearrecon_path = strcat('./recon_d=8/linearrecon/', patient_id, '/linearrecon_81_20iters.fld');

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
    
    fprintf('----- Calculating %s -----\n', patient_id);

    for mask_id = 1:length(mask_file_names)
        mask_file_name = mask_file_names{mask_id};
        mask_file_path = strcat('./masks/', patient_id, '/ROI/', mask_file_name);
        mask = load(mask_file_path).mask;
        
        mean_mask_s = mean(s(mask==1));
        mean_mask_f = mean(f(mask==1));
        mean_mask_p = mean(p(mask==1));
        mean_mask_l = mean(l(mask==1));

        sf = mean_mask_s / mean_mask_f;
        pf = mean_mask_p / mean_mask_f;
        lf = mean_mask_l / mean_mask_f;

        if contains(mask_file_name, 'lesion')

            RA_Lesion_sperf(end+1) = sf;
            RA_Lesion_partial(end+1) = pf;
            RA_Lesion_linear(end+1) = lf;        

        elseif contains(mask_file_name, 'KIDNEY')

            RA_Kidney_sperf(end+1) = sf;
            RA_Kidney_partial(end+1) = pf;
            RA_Kidney_linear(end+1) = lf;
            RA_All_Organs_sperf(end+1) = sf;
            RA_All_Organs_partial(end+1) = pf;
            RA_All_Organs_linear(end+1) = lf;            
        
        else
            fprintf("Something unexpected happend: %s!\n", mask_file_path);

        end
    end

end

%%

% Calculate the means of the arrays
Lesion_Means = [mean(RA_Lesion_sperf), mean(RA_Lesion_partial), mean(RA_Lesion_linear)];
Kidney_Means = [mean(RA_Kidney_sperf), mean(RA_Kidney_partial), mean(RA_Kidney_linear)];
% Create the table
RowTitles = {'Lesions', 'Kidneys'};
ColumnTitles = {'SpeRF', 'Partial', 'LinInt'};
DataMatrix = [Lesion_Means; Kidney_Means];
RATable = array2table(DataMatrix, 'VariableNames', ColumnTitles, 'RowNames', RowTitles);

% Display the table
disp(RATable);

RATableFilename = 'RA_table_d=8';

% Save the table to a .mat file
resultsPath = fullfile(save_ra_dir, RATableFilename);
save(resultsPath, 'RATable', ...
    'RA_Lesion_sperf', 'RA_Kidney_sperf', ...
    'RA_Lesion_partial', 'RA_Kidney_partial', ...
    'RA_Lesion_linear', 'RA_Kidney_linear', ...
    'RA_All_Organs_sperf', 'RA_All_Organs_partial', 'RA_All_Organs_linear');