function psma_linear_combine_two_beds(PSMA_patient_id)
    %% Automate patient name extraction
    patient_id = PSMA_patient_id;
    
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
    psma_linear_recon_bed1(patient_id, patient_name);
    psma_linear_recon_bed2(patient_id, patient_name);
    
    %% Combine two beds
    bed1_folder = ['./recon/linear_recon/psma_', patient_id, '_b1'];
    bed2_folder = ['./recon/linear_recon/psma_', patient_id, '_b2'];
    linear_bed1 = fld_read(fullfile(bed1_folder, 'linearrecon_79slices.fld'));
    linear_bed2 = fld_read(fullfile(bed2_folder, 'linearrecon_79slices.fld'));
    
    linear_2beds = cat(3, linear_bed1(:,:,:,:), linear_bed2(:,:,:,:));
    
    savefolder = ['./recon/linear_recon/psma_', patient_id];
    if ~exist(savefolder, 'dir')
        mkdir(savefolder)
    end
    
    fld_write(fullfile(savefolder, 'linearrecon_isaac_79+79_20iters.fld'), linear_2beds, 'type', 'xdr_float');
    fld_write(fullfile(savefolder, 'linearrecon_isaac_79+79.fld'), linear_2beds(:,:,:,10), 'type', 'xdr_float');
end