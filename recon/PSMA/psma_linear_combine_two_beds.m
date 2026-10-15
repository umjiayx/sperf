function psma_linear_combine_two_beds(PSMA_patient_id)
    patient_id = PSMA_patient_id;
    
    %% Recon!
    psma_linear_recon_bed1(patient_id);
    psma_linear_recon_bed2(patient_id);
    
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