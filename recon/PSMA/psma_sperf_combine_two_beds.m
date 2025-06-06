function psma_sperf_combine_two_beds(PSMA_patient_id)
    %% Automate patient name extraction
    patient_id = PSMA_patient_id; 
    start = 1;
    
    patternCl = '^cl\d+_(?<name>[a-z]+)_c\d_s\d$';
    patternNum = '^[01]\d*(?<name>[a-z]+)_c\d_s\d$';
    
    if regexp(patient_id, patternCl)
        patient_name = regexprep(upper(regexp(patient_id, patternCl, 'names').name), '^.*_', '');
    elseif regexp(patient_id, patternNum)
        patient_name = regexprep(upper(regexp(patient_id, patternNum, 'names').name), '^.*_', '');
    else
        error('Invalid format for patient_id.');
    end
    
    disp(['patient name: ', patient_name]);
    
    %% Recon!
    psma_sperf_recon_bed1(patient_id, patient_name, start);
    psma_sperf_recon_bed2(patient_id, patient_name, start);
    
    %% Combine two beds
    bed1_folder = strcat('./recon/sperf_recon/psma_', patient_id, '_b1');
    bed2_folder = strcat('./recon/sperf_recon/psma_', patient_id, '_b2');
    sperf_bed1 = fld_read(strcat(bed1_folder, '/sperfrecon_79slices.fld'));
    sperf_bed2 = fld_read(strcat(bed2_folder, '/sperfrecon_79slices.fld'));

    sperf_2beds = cat(3, sperf_bed1(:,:,:,:), sperf_bed2(:,:,:,:));
    
    savefolder = strcat('./recon/sperf_recon/psma_', patient_id);
    if ~exist(savefolder, 'dir')
        mkdir(savefolder)
    end

    fld_write(fullfile(savefolder, 'sperfrecon_isaac_79+79_20iters.fld'), sperf_2beds, 'type', 'xdr_float');
    fld_write(fullfile(savefolder, 'sperfrecon_isaac_79+79.fld'), sperf_2beds(:,:,:,10), 'type', 'xdr_float');
end